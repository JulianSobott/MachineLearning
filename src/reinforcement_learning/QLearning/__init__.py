"""
https://en.wikipedia.org/wiki/Q-learning

Q-learning is a model-free reinforcement learning algorithm. The goal of Q-learning is to learn a policy, which tells
an agent what action to take under what circumstances. It does not require a model (hence the connotation
"model-free") of the environment, and it can handle problems with stochastic transitions and rewards,
without requiring adaptations.

For any finite Markov decision process (FMDP), Q-learning finds a policy that is optimal in the sense that it
maximizes the expected value of the total reward over any and all successive steps, starting from the current state.[
1] Q-learning can identify an optimal action-selection policy for any given FMDP, given infinite exploration time and
a partly-random policy.[1] "Q" names the function that returns the reward used to provide the reinforcement and can
be said to stand for the "quality" of an action taken in a given state.

S: states
A: Actions per state
F(s, a) = {s_1', s2', ... }

Q: S x A -> R

Q_new(s_t, a_t) <- (1 - alpha) * Q(s_t, a_t) + alpha * (r_t + discountFactor * max(Q(s_{t+1}, a)))

newValue        <-               oldValue  learningRate reward                estimateOfOptimalFutureValue


Implementation variants:
-------------------------

Q-learning at its simplest stores data in tables. This approach falters with increasing numbers of states/actions.

**Function approximation:** Using an artificial neural network. For large problems. State space can be continuous.

**Quantization:** Vectors describe a state. Shrink possible actions to buckets. (e.g. vector: (velocity, position))
"""
