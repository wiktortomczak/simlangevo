#!/usr/bin/env python

"""Simulation of language evolution.

See https://docs.google.com/document/d/1HMq4m5xd_ggTzM7XJiXOSbDRTT2jQ5foSlB2aZEw6GM
"""

# Configurations reported in the doc, in the Results section.
# Note default command-line flag values, in flag definitions below.
#
# # Configuration #1
#
# ./simlangevo.py  \
#   --agent=fnn-obverter  \
#   --agent_training=supervised  \
#   --meanings=OneHot:3  \
#   --messages=Binary:3 \
#   --num_generations=121
#
# # Configuration #2
#
# ./simlangevo.py  \
#   --agent=fnn-obverter  \
#   --agent_training=supervised  \
#   --meanings=Binary:3  \
#   --messages=Binary:3  \
#   --num_generations=151
#
# # Configuration #3
#
# ./simlangevo.py  \
#   --agent=policy-fnn-obverter  \
#   --agent_training=reinforcement  \
#   --meanings=Discrete:3  \
#   --messages=Binary:3  \
#   --num_agents=10  \
#   --nn_learning_rate=0.01  \
#   --num_generations=201
#
# # Configuration #4
#
# ./simlangevo.py  \
#   --agent=policy-fnn-obverter  \
#   --agent_training=reinforcement  \
#   --meanings=Discrete:8  \
#   --messages=Binary:3  \
#   --num_agents=10  \
#   --nn_learning_rate=0.01  \
#   --num_generations=501

from __future__ import division

import collections
import random
import sys

import gflags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# Command-line flag definitions. Agent / simulation / environment parameters.
# Roughly correspond to the Parameters section in the doc.
gflags.DEFINE_string('agent', 'fnn-obverter', '')
gflags.DEFINE_string('agent_training', 'supervised', '')
gflags.DEFINE_string('agent_replacement', 'none', '')
gflags.DEFINE_string('meanings', 'OneHot:3', '')
gflags.DEFINE_string('messages', 'Binary:3', '')
gflags.DEFINE_integer('num_agents', 100, '')
gflags.DEFINE_integer('num_generations', 501, '')
gflags.DEFINE_integer('num_generations_between_metrics', 10, '')
gflags.DEFINE_integer('num_communication_acts', 100, '')
gflags.DEFINE_integer('nn_hidden_layer_size', 8, '')
gflags.DEFINE_float('nn_learning_rate', .1, '')
gflags.DEFINE_boolean('record_language', True, '')
gflags.DEFINE_list('metrics', ['communication_accuracy:1000'], '')

FLAGS = gflags.FLAGS


def main(argv):
  """Executable entry point. Sets up and runs a simulation of lang evolution."""
  argv = FLAGS(argv)  # Parse command-line flags.

  # Fix random seed for reproducibility.
  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  # Instantiate meaning and message spaces.
  global MEANINGS, MESSAGES  # TODO: Clean up.
  MEANINGS = MeaningSpace.FromFlags()
  MESSAGES = MessageSpace.FromFlags()

  # Run the simulation.
  language, metrics = (
    IteratedLearning.FromFlags().
    Run(FLAGS.num_generations, FLAGS.record_language))
  
  # Print / plot metrics and other simulation result summaries.
  print metrics.to_string()
  metrics.plot()
  plt.show()
  if language is not None:
    PlotDialectMaps(language)
    plt.show()


class Agent(object):
  """An agent that communicates with other agents and learns a shared language.

  Abstract interface.
  """

  @classmethod
  def FromFlags(cls):
    """Creates an instance specified by flags."""
    return cls._AGENT_REGISTRY[FLAGS.agent].FromFlags()

  def Speak(self, meaning):
    """Makes the agent say / produce a message, the one it maps the meaning to.

    Args:
      meaning: element of MeaningSpace. Meaning to say.
    Returns:
      Element of MessageSpace. Message encoding the meaning.
    """
    raise NotImplementedError

  def Hear(self, message):
    """Makes the agent hear / recover a meaning, the one it maps the message to.

    Args:
      message: element of MessageSpace. Received message.
    Returns:
      Element of MeaningSpace. Recovered meaning.
    """
    raise NotImplementedError

  # Registry of Agent implementations (subclasses).
  # Human-readable string Agent class specification -> Agent subclass.
  _AGENT_REGISTRY = {}


class HebbianAgent(Agent):
  """An agent with a 2D meaning x message matrix, trained with Hebbian rule.

  The matrix holds the (meaning, message) association strength for each pair.
  It specifies both meaning -> message and message -> meaning mappings:
    meaning -> message (Speak): the agent looks at the matrix row corresponding
      to the meaning and takes the message (column) with the highest value.
    message -> meaning (Hear): the agent looks at the matrix column corresponding
      to the message and takes the meaning (row) with the highest value.

    If the highest value is tied between more than one column / row (more than
    one candidate), one is picked at random.

  Learning consists in strengthening the association for a given
  (meaning, message) pair, by increasing the number in the appropriate matrix
  cell, and optionally weaking alternative associations, by decreasing other
  numbers in the same column and row ("lateral inhibition").

  Source: github.com/smkirby/SimLang/blob/master/simlang_13_lab.ipynb
  """

  @classmethod
  def FromFlags(cls):
    """See base class."""
    meaning_message_associations = (
      np.zeros((len(MEANINGS), len(MESSAGES)), dtype=np.uint))
    return cls(meaning_message_associations)

  def __init__(self, meaning_message_associations):
    """Constructor.

    Args:
      meaning_message_associations: np.NDArray, rows: meanings, cols: messages.
    """
    self._meaning_message_associations = meaning_message_associations

  def Speak(self, meaning):
    """See base class."""
    return _RandomMaxIndex(self._meaning_message_associations[meaning])

  def Hear(self, message):
    """See base class."""
    return _RandomMaxIndex(self._meaning_message_associations[:, message])

  # TODO: Adapt to AgentTraining interface:
  #   def LearnUnsupervised(self, speaker_meaning, message)
  def Train(self, meaning_message_pairs, num_epochs):
    for _ in range(num_epochs):
      for __ in enumerate(meaning_message_pairs):
        meaning, message = _RandomChoice(meaning_message_pairs)
        self._meaning_message_associations[meaning, message] += 1
        # TODO: inhibition

Agent._AGENT_REGISTRY['hebbian'] = HebbianAgent

def _RandomMaxIndex(arr):
  """Returns the index of the maximum value in the array.

  If there are multiple maximum values, returns the index of one of them
  at random.

  Args:
    arr: np.NDArray.
  Returns:
    int. The index.
  """
  max_indices = np.argwhere(arr == arr.max()).flatten()
  if len(max_indices) == 1:
    return max_indices[0]
  else:
    return _RandomChoice(max_indices)

  
class FeedForwardNetworkObverterAgent(Agent):
  """An agent with a feed-forward neural network decoding message -> meaning.

  meaning -> message mapping is derived from message -> meaning mapping via
  obverter procedure (inverting the latter by evaluating all possible messages
  and taking the message that produces the closest meaning).

  The message -> meaning network is trained in supervised fashion, from
  (speaker / intended meaning, message, hearer / recovered meaning) triplets,
  acting as (target output, input, predicted output) respectively. The agent
  directly improves its capacity to hear, and indirectly its capacity to speak
  (by obverting the improved hearing network).

  The agent assumes structured messages and meanings, each consisting of
  a number of bits that map to input and output neurons, respectively.
  The network learns / discovers this structure.

  Source:
    feed-forward network: TODO.
    obverter: TODO.
  """

  @classmethod
  def FromFlags(cls):
    """See base class."""
    # Instantiate network architecture, possibly including a hidden layer,
    # depending on --nn_hidden_layer_size.
    if not FLAGS.nn_hidden_layer_size:  # layers: input, output
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, MEANINGS.num_bits),
        torch.nn.Sigmoid()
      )
    else:  # layers: input, hidden, output
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, FLAGS.nn_hidden_layer_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(FLAGS.nn_hidden_layer_size, MEANINGS.num_bits),
        torch.nn.Sigmoid()
      )
    optimizer = torch.optim.Adam(
      message_to_meaning_fnn.parameters(), FLAGS.nn_learning_rate)
    return cls(message_to_meaning_fnn, optimizer)

  def __init__(self, message_to_meaning_fnn, optimizer):
    """Constructor.

    Args:
      message_to_meaning_fnn: torch.nn.Module. Message -> meaning decoder.
      optimizer: torch.optim.Optim. Network training, bound to the above object.
    """
    self._message_to_meaning_fnn = message_to_meaning_fnn
    self._optimizer = optimizer

  def Speak(self, meaning):
    """See base class."""
    with torch.no_grad():
      # Obverter: To say given meaning, the speaker picks the message that
      # he himself associates most closely with (maps to) the meaning
      # (the speaker inverts his message -> meaning mapping).
      # TOOD: Optimize, eg. lookup table?
      return min(MESSAGES, key=lambda message: MEANINGS.Distance(
        self._MessageToMeaning(message), meaning))

  def Hear(self, message):
    """See base class."""
    with torch.no_grad():
      return _ToBinary(self._MessageToMeaning(message))

  def LearnSupervised(self, speaker_meaning, message, hearer_meaning):
    """See base class."""  # TOOD: SupervisedLearnerMixin.
    # self is hearer, the agent that mapped message to hearer_meaning.
    # In supervised learning terms:
    #   message - x
    #   hearer_meaning - y_hat (predicted)
    #   speaker_meaning - y (target)
    try:
      self._message_to_meaning_fnn.train()
      # Original hearer_meaning, real numbers from [0, 1] not binary,
      # not detached from the computation graph (allowing to compute gradient
      # of network weights).
      # TODO: Retain at source, in Hear().
      hearer_meaning = self._MessageToMeaning(message, detach=False)
      # Wrap in tensor, needed for further operations on hearer_meaning.
      speaker_meaning = torch.tensor(speaker_meaning, dtype=torch.uint8)
      loss = MEANINGS.Distance(speaker_meaning, hearer_meaning)
      self._optimizer.zero_grad()
      loss.backward()
      self._optimizer.step()
    finally:
      self._message_to_meaning_fnn.eval()
    
  def _MessageToMeaning(self, message, detach=True):
    """Maps given message to meaning. Handles PyTorch network invocation.

    Args:
      message: element of MessageSpace. Message to map.
      detach: bool. If False, the meaning is not detached from PyTorch
        computation graph, allowing to use the meaning tensor for training.
    Returns:
      Element of MeaningSpace. Meaning mapped to message.
    """
    message_mb = torch.FloatTensor(message.reshape(1, -1))  # Add batch dim.
    meaning_mb = self._message_to_meaning_fnn(message_mb)
    meaning = meaning_mb.view(-1)  # Drop batch dimension.
    if detach:
      return meaning.detach().numpy()
    else:
      return meaning

Agent._AGENT_REGISTRY['fnn-obverter'] = FeedForwardNetworkObverterAgent

def _ToBinary(arr):
  """Binarizes given array with numerical values to binary 0-1 values."""
  return (arr >= .5).astype(np.uint8)


class PolicyFeedForwardNetworkObverterAgent(Agent):
  """An agent with a feed-forward policy network p(meaning|message) for decoding.

  meaning -> message mapping is derived from message -> meaning via obverter;
  see FeedForwardNetworkObverterAgent for details.

  The message -> meaning policy network is trained via REINFORCE algorithm,
  from (message, hearer / recovered meaning, reward) triplets.

  The agent assumes structured messages, each consisting of a number of bits
  that map to input neurons. The output layer has one neuron for each possible
  meaning, which is transformed to meaning probabilities with a softmax.

  Source:
    policy network, REINFORCE: TODO.
    obverter: TODO.
  """

  @classmethod
  def FromFlags(cls):
    """See base class."""
    # Instantiate network architecture, possibly including a hidden layer,
    # depending on --nn_hidden_layer_size.
    if not FLAGS.nn_hidden_layer_size:  # layers: input, output
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, len(MEANINGS))
      )
    else:  # layers: input, hidden, output
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, FLAGS.nn_hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(FLAGS.nn_hidden_layer_size, len(MEANINGS))
      )
    optimizer = torch.optim.Adam(
      message_to_meaning_fnn.parameters(), FLAGS.nn_learning_rate)
    return cls(message_to_meaning_fnn, optimizer)

  def __init__(self, message_to_meaning_fnn, optimizer):
    """Constructor.

    Args:
      message_to_meaning_fnn: torch.nn.Module. Message -> meaning decoder.
      optimizer: torch.optim.Optim. Network training, bound to the above object.
    """
    self._message_to_meaning_fnn = message_to_meaning_fnn
    self._optimizer = optimizer

  def Speak(self, meaning):
    """See base class."""
    meaning_id = Iterables.Index(MEANINGS, meaning)
    with torch.no_grad():
      # Obverter: To say given meaning, evaluate all messages and use the
      # message for which the score for the given meaning is the highest.
      return max(MESSAGES, key=lambda message: (
        self._MessageToMeaningScore(message)[meaning_id]))

  def Hear(self, message):
    """See base class."""
    with torch.no_grad():
      # Compute meaning distribution.
      meaning_score = self._MessageToMeaningScore(message)
    # Sample a meaning from computed distribution.
    meaning_proba = F.softmax(meaning_score, dim=0)
    return np.random.choice(MEANINGS, p=meaning_proba.numpy())

  def _MessageToMeaningScore(self, message):
    """Computes score distribution over possible meanings, given message.

    Handles PyTorch network invocation.

    Args:
      message: element of MessageSpace. Message to compute meaning scores for.
    Returns:
      torch.FloatTensor. Score distribution over meanings.
    """
    message = torch.FloatTensor(message.reshape(1, -1))
    return self._message_to_meaning_fnn(message).view(-1).detach()

  def LearnToHearReinforced(self, messages, hearer_meanings, rewards):
    """See base class."""  # TOOD: ReinforcementLearnerMixin.
    # self is hearer, the agent that mapped message to hearer_meaning.
    # In reinforcement learning terms:
    #   message - state / observation
    #   hearer_meaning - agent's action in response to messsage
    n = len(messages)
    assert n == len(messages) == len(hearer_meanings) == len(rewards)

    self._message_to_meaning_fnn.train()

    # REINFORCE algorithm.
    meaning_scores = self._message_to_meaning_fnn(torch.FloatTensor(messages))
    meaning_log_probas = F.log_softmax(meaning_scores, dim=1)
    meaning_ids = [Iterables.Index(MEANINGS, m) for m in hearer_meanings]
    selected_meaning_log_probas = meaning_log_probas[np.arange(n), meaning_ids]
    loss = -(torch.FloatTensor(rewards) * selected_meaning_log_probas).mean()
    
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

    self._message_to_meaning_fnn.eval()

Agent._AGENT_REGISTRY['policy-fnn-obverter'] = PolicyFeedForwardNetworkObverterAgent


# TODO: Clean up.
class AgentTraining(object):
  """Agent training method. Calls relevant Agent method, eg. LearnSupervised."""

  @classmethod
  def FromFlags(cls):
    if FLAGS.agent_training == 'supervised':
      return AgentSupervisedTraining()
    elif FLAGS.agent_training == 'unsupervised':
      return AgentUnsupervisedTraining()
    elif FLAGS.agent_training == 'reinforcement':
      return AgentReinforcementTraining()
    else:
      raise ValueError(FLAGS.agent_training)

  def TrainAgents(self, communication_act):
   raise NotImplementedError

class AgentSupervisedTraining(AgentTraining):

  def TrainAgents(self, communication_act):
   communication_act.hearer.LearnSupervised(
     communication_act.speaker_meaning,
     communication_act.message,
     communication_act.hearer_meaning)

class AgentReinforcementTraining(AgentTraining):

  def TrainAgents(self, communication_acts):
    # for speaker, speaker_acts in Iterables.GroupBy(communication_acts, lambda ca: ca.speaker):
    #   speaker.LearnToSpeakReinforced(
    #     [sa.speaker_meaning for sa in speaker_acts],
    #     [sa.message for sa in speaker_acts],
    #     [self._ComputeReward(sa) for sa in speaker_acts])
    for hearer, hearer_acts in Iterables.GroupBy(communication_acts, lambda ca: ca.hearer):
      hearer.LearnToHearReinforced(
        [ha.message for ha in hearer_acts],
        [ha.hearer_meaning for ha in hearer_acts],
        [self._ComputeReward(ha) for ha in hearer_acts])

  @staticmethod
  def _ComputeReward(communication_act):
    return int(MEANINGS.Equal(
      communication_act.speaker_meaning, communication_act.hearer_meaning))


class Space(object):
  """A space of objects, as in mathematics. Abstract interface."""

  @classmethod
  def Create(cls, spec):
    """Creates an instance from a string specification.

    Args:
      spec: str. Space specificiation.
    Returns:
      Space instance.
    """
    args = spec.split(':')
    if args[0] == 'Binary':
      num_bits = int(args[1])
      return BinaryVectorSpace(np.array([
        map(int, np.binary_repr(n, num_bits))
        for n in xrange(2**num_bits)], dtype=np.uint8))
    elif args[0] == 'OneHot':
      num_bits = int(args[1])
      return BinaryVectorSpace(np.eye(num_bits, dtype=np.uint8))
    elif args[0] == 'Discrete':
      num_elements = int(args[1])
      return DiscreteSpace(np.arange(num_elements))
    else:
      raise ValueError(spec)

  # __len__ and __getiem__ allow iterating and samlping.

  def __len__(self):
    raise NotImplementedError

  def __getitem__(self, i):
    raise NotImplementedError

class BinaryVectorSpace(Space):
  """Space of binary vectors {0,1}^n of fixed number of bits n."""

  def __init__(self, vectors):
    """Constructor.

    Args:
      vectors: np.NDArray, 2D. Rows are vectors in the space.
    """
    self._vectors = vectors

  def __len__(self):
    return len(self._vectors)

  def __getitem__(self, i):
    return self._vectors[i]

  @property
  def num_bits(self):
    return self._vectors.shape[1]

  @staticmethod
  def Distance(a, b):
    """Computes distance between two vectors.

    Args:
      a, b: binary vectors. Elements of the space.
    Returns:
      float.
    """
    return ((a - b)**2).mean()

  @staticmethod
  def Equal(a, b):
    """Whether two vectors are equal.

    Args:
      a, b: binary vectors. Elements of the space.
    Returns:
      bool.
    """
    return (a == b).all()

class DiscreteSpace(Space):
  """Space of distinct discrete symbols with abstract / unspecified semantics."""

  def __init__(self, symbols):
    """Constructor.

    Args:
      symbols: array-like of symbols.
    """
    self._symbols = symbols

  def __len__(self):
    return len(self._symbols)

  def __getitem__(self, i):
    return self._symbols[i]

  @staticmethod
  def Equal(a, b):
    """Whether two symbols are equal.

    Args:
      a, b: symbols. Elements of the space.
    Returns:
      bool.
    """
    return a == b


class MeaningSpace(Space):
  """Meaning space. All possible meanings that agents might refer to."""

  @classmethod
  def FromFlags(cls):
    """Creates an instance specified by flags."""
    return cls.Create(FLAGS.meanings)
                        
class MessageSpace(Space):
  """Message space. All possible messages that agents might say."""

  @classmethod
  def FromFlags(cls):
    """Creates an instance specified by flags."""
    return cls.Create(FLAGS.messages)


class IteratedLearning(object):
  """Simulation of language evolution algorithm / main loop."""

  @classmethod
  def FromFlags(cls):
    """Creates an instance specified by flags."""
    agent_factory = Agent.FromFlags
    agents = [agent_factory() for _ in range(FLAGS.num_agents)]
    return cls(
      agents,
      AgentTraining.FromFlags(),
      AgentReplacementStrategy.FromFlags(agent_factory),
      FLAGS.num_communication_acts,
      Metric.FromFlags())

  def __init__(self,
               agents, agent_training, agent_replacement_strategy,
               num_communication_acts,
               metrics):
    """Constructor.

    Args:
      agents: list of Agent. Communicate, learn, get replaced between generations.
      agent_training: AgentTraining. Training method. Must be supported by agents.
      agent_replacement_strategy: AgentReplacementStrategy.
      num_communication_acts: int. Number of communication acts per generation
        to sample / generate.
      metrics: list of Metric. Metrics to compute every few generations.
    """
    self._agents = agents
    self._agent_training = agent_training
    self._agent_replacement_strategy = agent_replacement_strategy
    self._num_communication_acts = num_communication_acts
    self._metrics = metrics

  def Run(self, num_generations, record_language):
    """Runs the simulation main loop.

    Args:
      num_generations: int. Number of generations to run the simulation for.
    Returns:
      language: np.NDArray (generations x meanings x agents x message bits),
      metrics: pd.DataFrame, columns: metrics, index: generation ids
    """
    # Allocate placeholders for return values.
    if record_language:
      language = np.empty(
        (num_generations, len(MEANINGS), len(self._agents), MESSAGES.num_bits),
        dtype=np.uint8)
    else:
      language = None
    metrics_per_generation = []

    for generation_id in xrange(num_generations):
      # Have agents communicate in pairs. Generate communication samples.
      communication_acts = [
        self._DoCommunicationAct()
        for _ in xrange(self._num_communication_acts)]

      # Have each agent learn from his samples.
      self._agent_training.TrainAgents(communication_acts)

      # Update agent population.
      self._agents, new_agents = (
        self._agent_replacement_strategy.AddRemoveAgents(self._agents))
      # TODO: Train new agents?

      # Record language.
      if record_language:
        for meaning_id, meaning in enumerate(MEANINGS):
          for agent_id, agent in enumerate(self._agents):
            language[generation_id, meaning_id, agent_id] = agent.Speak(meaning)

      # Compute metrics.
      if generation_id % FLAGS.num_generations_between_metrics == 0:
        metrics = self._ComputeMetrics()
        print 'generation_id=%u %s' % (generation_id, Dicts.ToString(metrics))
        metrics_per_generation.append(
          Dicts.Merge(dict(generation_id=generation_id), metrics))

    metrics_df = pd.DataFrame(metrics_per_generation).set_index('generation_id')
    return language, metrics_df

  def _DoCommunicationAct(self):
    """Makes an agent communicate a meaning to another agent via a message.

    Samples a speaker agent, a hearer agent and a meaning. Makes the speaker
    say the meaning and the hearer recover a meaning. Records these 5 pieces.

    Returns:
      CommunicationAct. A record of the above communication
    """
    speaker, hearer = _RandomChoice(self._agents, 2, replace=False)
    speaker_meaning = _RandomChoice(MEANINGS)  # TODO: Subset of meanings, agent choice.
    message = speaker.Speak(speaker_meaning)
    hearer_meaning = hearer.Hear(message)
    return CommunicationAct(
      speaker, hearer, speaker_meaning, message, hearer_meaning)

  def _ComputeMetrics(self):
    return {
      metric.name: metric.Compute(self._agents)
      for metric in self._metrics}

"""A record of the following communication act:

meaning  -- speaker.Speak(.) -->  message  -- hearer.Hear(.)  -->  meaning
"""
CommunicationAct = collections.namedtuple('CommunicationAct', [
  'speaker', 'hearer', 'speaker_meaning', 'message', 'hearer_meaning'])


class AgentReplacementStrategy(object):

  @classmethod
  def FromFlags(cls, agent_factory_func):
    if FLAGS.agent_replacement == 'all':
      cls = ReplaceAll
    elif FLAGS.agent_replacement == 'one':
      cls = ReplaceOne
    elif FLAGS.agent_replacement == 'none':
      cls = ReplaceNone
    else:
      raise ValueError(FLAGS.agent_replacement)
    return cls(agent_factory_func)

  def __init__(self, agent_factory_func):
    self._agent_factory_func = agent_factory_func

  def AddRemoveAgents(self, agents):
    raise NotImplementedError

  def _CreateAgent(self):
    return self._agent_factory_func()

class ReplaceAll(AgentReplacementStrategy):

  def AddRemoveAgents(self, agents):
    new_agents = [self._CreateAgent() for _ in enumerate(agents)]
    return new_agents, new_agents

class ReplaceOne(AgentReplacementStrategy):

  def AddRemoveAgents(self, agents):
    new_agent = self._CreateAgent()
    return agents[1:] + [new_agent], [new_agent]

class ReplaceNone(AgentReplacementStrategy):

  def AddRemoveAgents(self, agents):
    return agents, agents


class Metric(object):

  @classmethod
  def FromFlags(cls):
    return [cls.FromSpec(metric_spec) for metric_spec in FLAGS.metrics]

  @classmethod
  def FromSpec(cls, spec):
    metric_name = spec.split(':')[0]
    if metric_name == 'communication_accuracy':
      return CommunicationAccuracy.FromSpec(spec)
    else:
      raise ValueError(spec)

  @property
  def name(self):
    raise NotImplementedError

  def Compute(self, agents):
    raise NotImplementedError


class CommunicationAccuracy(Metric):

  @classmethod
  def FromSpec(cls, spec):
    args = spec.split(':')
    assert args[0] == 'communication_accuracy'  # TODO: Clean up.
    num_samples = int(args[1])
    return cls(num_samples)

  def __init__(self, num_samples):
    self._num_samples = num_samples

  @property
  def name(self):
    return 'communication_accuracy'

  def Compute(self, agents):
    num_successful = 0
    for _ in xrange(self._num_samples):
      hearer, speaker = _RandomChoice(agents, 2, replace=False)
      speaker_meaning = _RandomChoice(MEANINGS)
      hearer_meaning = hearer.Hear(speaker.Speak(speaker_meaning))
      num_successful += int(MEANINGS.Equal(speaker_meaning, hearer_meaning))
    return num_successful / self._num_samples
 

def PlotDialectMaps(language):
  # language: generations x meanings x agents x message bits
  dialect_maps = [
    CreateDialectMap(meaning)
    for meaning in np.swapaxes(language, 0, 1)]

  fig, axs = plt.subplots(ncols=len(MEANINGS))
  for meaning_id, (ax, dialect_map) in enumerate(zip(axs, dialect_maps)):
    ax.set_title('meaning #%u' % meaning_id)
    ax.imshow(dialect_map)
    ax.axes.get_xaxis().set_visible(False)

def CreateDialectMap(meaning):
  # meaning: generations x agents x message bits.
  assert meaning.ndim == 3
  assert meaning.shape[2] == 3
  # Map {0, 1} to {0, 255} = map 3-bit messages to RGB triplets in 0-255 range.
  # Otherwise, nothing to do, meaning shape is already (M, N, 3), as expected
  # by imshow().
  return meaning * 255

  
class Dicts(object):

  @classmethod
  def Merge(cls, *dicts):
    result = {}
    for d in dicts:
      for k, v in d.iteritems():
        result[k] = v
    return result

  @classmethod
  def ToString(cls, d):
    # TODO: Formatting.
    return ' '.join(['%s=%s' % (k, v) for k, v in sorted(d.items())])


class Iterables(object):
        
  @classmethod
  def Index(cls, iterable, key):
    for i, e in enumerate(iterable):
      if e == key:
        return i
    raise KeyError(key)

  @classmethod
  def GroupBy(cls, iterable, key, sorted=False):
    if not sorted:
      key_to_elements = collections.defaultdict(list)
      for e in iterable:
        key_to_elements[key(e)].append(e)
      return key_to_elements.iteritems()
    else:
      raise NotImplementedError  # TODO: itertools.groupby.


def _RandomChoice(arr, size=None, replace=True):
  if isinstance(arr, np.ndarray):
    return np.random.choice(arr, size, replace)
  else:
    if replace:
      assert size is None
      i = np.random.choice(len(arr))
      return arr[i]
    else:
      return random.sample(arr, size)


if __name__ == '__main__':
  main(sys.argv)
