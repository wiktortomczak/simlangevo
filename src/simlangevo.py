#!/usr/bin/env python

# ./simlangevo.py  --agent=policy-fnn-obverter  --agent_training=reinforcement  --meanings=Discrete:3  --messages=Binary:3  --num_agents=10  --nn_learning_rate=0.01   --num_generations=201

# ./simlangevo.py  --agent=policy-fnn-obverter  --agent_training=reinforcement  --meanings=Discrete:8  --messages=Binary:3  --num_agents=10  --nn_learning_rate=0.01   --num_generations=501

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
  argv = FLAGS(argv)

  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  global MEANINGS, MESSAGES  # TODO: Clean up.
  MEANINGS = MeaningSpace.FromFlags()
  MESSAGES = MessageSpace.FromFlags()

  language, metrics = (
    IteratedLearning.FromFlags().
    Run(FLAGS.num_generations, FLAGS.record_language))

  print metrics.to_string()
  metrics.plot()
  plt.show()
  if language is not None:
    PlotDialectMaps(language)
    plt.show()


class Agent(object):

  @classmethod
  def FromFlags(cls):
    return cls._AGENT_REGISTRY[FLAGS.agent].FromFlags()

  def Speak(self, meaning):
    raise NotImplementedError

  def Hear(self, message):
    raise NotImplementedError

  _AGENT_REGISTRY = {}


class HebbianAgent(Agent):

  @classmethod
  def FromFlags(cls):
    return cls()

  def __init__(self):
    self._meaning_message_weights = (
      np.zeros((len(MEANINGS), len(MESSAGES)), dtype=np.uint))

  def Speak(self, meaning):
    return _RandomMaxIndex(self._meaning_message_weights[meaning])

  def Hear(self, message):
    return _RandomMaxIndex(self._meaning_message_weights[:, message])

  # TODO: Adapt to AgentTraining interface:
  #   def LearnUnsupervised(self, speaker_meaning, message)
  def Train(self, meaning_message_pairs, num_epochs):
    for _ in range(num_epochs):
      for __ in enumerate(meaning_message_pairs):
        meaning, message = _RandomChoice(meaning_message_pairs)
        self._meaning_message_weights[meaning, message] += 1
        # TODO: inhibition

Agent._AGENT_REGISTRY['hebbian'] = HebbianAgent

def _RandomMaxIndex(arr):
  max_indices = np.argwhere(arr == arr.max()).flatten()
  if len(max_indices) == 1:
    return max_indices[0]
  else:
    return _RandomChoice(max_indices)

  
class FeedForwardNetworkObverterAgent(Agent):

  @classmethod
  def FromFlags(cls):
    if not FLAGS.nn_hidden_layer_size:
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, MEANINGS.num_bits),
        torch.nn.Sigmoid()
      )
    else:
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
    self._message_to_meaning_fnn = message_to_meaning_fnn
    self._optimizer = optimizer

  def Speak(self, meaning):
    with torch.no_grad():
      # Obverter: To say given meaning, the speaker picks the message that
      # he himself associates most closely with (maps to) the meaning
      # (the speaker inverts his message -> meaning mapping).
      # TOOD: Optimize, eg. lookup table?
      return min(MESSAGES, key=lambda message: MEANINGS.Distance(
        self._MessageToMeaning(message), meaning))

  def Hear(self, message):
    with torch.no_grad():
      return _ToBinary(self._MessageToMeaning(message))

  def LearnSupervised(self, speaker_meaning, message, hearer_meaning):
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
    message_mb = torch.FloatTensor(message.reshape(1, -1))  # Add batch dim.
    meaning_mb = self._message_to_meaning_fnn(message_mb)
    meaning = meaning_mb.view(-1)  # Drop batch dimension.
    if detach:
      return meaning.detach().numpy()
    else:
      return meaning

Agent._AGENT_REGISTRY['fnn-obverter'] = FeedForwardNetworkObverterAgent

def _ToBinary(arr):
  return (arr >= .5).astype(np.uint8)


class PolicyFeedForwardNetworkObverterAgent(Agent):

  @classmethod
  def FromFlags(cls):
    if not FLAGS.nn_hidden_layer_size:
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, len(MEANINGS))
      )
    else:
      message_to_meaning_fnn = torch.nn.Sequential(
        torch.nn.Linear(MESSAGES.num_bits, FLAGS.nn_hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(FLAGS.nn_hidden_layer_size, len(MEANINGS))
      )
    optimizer = torch.optim.Adam(
      message_to_meaning_fnn.parameters(), FLAGS.nn_learning_rate)
    return cls(message_to_meaning_fnn, optimizer)

  def __init__(self, message_to_meaning_fnn, optimizer):
    self._message_to_meaning_fnn = message_to_meaning_fnn
    self._optimizer = optimizer

  def Speak(self, meaning):
    meaning_id = Iterables.Index(MEANINGS, meaning)
    with torch.no_grad():
      return max(MESSAGES, key=lambda message: (
        self._MessageToMeaningScore(message)[meaning_id]))

  def Hear(self, message):
    with torch.no_grad():
      meaning_score = self._MessageToMeaningScore(message)
    return np.random.choice(MEANINGS, p=F.softmax(meaning_score, dim=0).numpy())

  def _MessageToMeaningScore(self, message):
    message = torch.FloatTensor(message.reshape(1, -1))
    return self._message_to_meaning_fnn(message).view(-1).detach()

  def LearnToHearReinforced(self, messages, hearer_meanings, rewards):
    self._message_to_meaning_fnn.train()
    # self is hearer, the agent that mapped message to hearer_meaning.
    # In reinforcement learning terms:
    #   message - state / observation
    #   hearer_meaning - agent's action in response to messsage
    n = len(messages)
    assert n == len(messages) == len(hearer_meanings) == len(rewards)

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


class AgentTraining(object):

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

  @classmethod
  def Create(cls, spec):
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

  def __len__(self):
    raise NotImplementedError

  def __getitem__(self, i):
    raise NotImplementedError

class BinaryVectorSpace(Space):

  def __init__(self, vectors):
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
    return ((a - b)**2).mean()

  @staticmethod
  def Equal(a, b):
    return (a == b).all()

class DiscreteSpace(Space):

  def __init__(self, elements):
    self._elements = elements

  def __len__(self):
    return len(self._elements)

  def __getitem__(self, i):
    return self._elements[i]

  @staticmethod
  def Equal(a, b):
    return a == b


class MeaningSpace(Space):
  @classmethod
  def FromFlags(cls):
    return cls.Create(FLAGS.meanings)
                        
class MessageSpace(Space):
  @classmethod
  def FromFlags(cls):
    return cls.Create(FLAGS.messages)


class IteratedLearning(object):

  @classmethod
  def FromFlags(cls):
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
    self._agents = agents
    self._agent_training = agent_training
    self._agent_replacement_strategy = agent_replacement_strategy
    self._num_communication_acts = num_communication_acts
    self._metrics = metrics

  def Run(self, num_generations, record_language):
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


def _DumpArray(arr, fmt='%u'):
  np.savetxt(sys.stdout, arr, fmt=fmt)


if __name__ == '__main__':
  main(sys.argv)
