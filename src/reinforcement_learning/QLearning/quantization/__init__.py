"""
https://en.wikipedia.org/wiki/Q-learning#Quantization

Another technique to decrease the state/action space quantizes possible values. Consider the example of learning to
balance a stick on a finger. To describe a state at a certain point in time involves the position of the finger in
space, its velocity, the angle of the stick and the angular velocity of the stick. This yields a four-element vector
that describes one state, i.e. a snapshot of one state encoded into four values. The problem is that infinitely many
possible states are present. To shrink the possible space of valid actions multiple values can be assigned to a
bucket. The exact distance of the finger from its starting position (-Infinity to Infinity) is not known, but rather
whether it is far away or not (Near, Far).
"""