# =======================================================
# Hidden Markov Models with Gaussian emissions
# =======================================================
# Ning Ma, University of Sheffield 
# n.ma@sheffield.ac.uk

import numpy as np
from sklearn.mixture import GaussianMixture

class HMM:
    """ A left-to-right no-skip Hidden Markov Model with Gaussian emissions
    
    The states in this kind of model are linearly connected. Speech starts at one point
    and starts at another point.
    
    The front-end produces a stream of feature vectors (observation vectors)
    O = [o_1,...,o_t,...,o_T]
        
    The best word sequence W is the value of W that maximizes P(O|W)*P(W)
    
    It uses a GMM for the output distribution.
    It has Transition probability matrix A where a[i,j] is the (log) probability of going from state q_i to state q_j. It is a square n by n matrix
    It has mean vectors \mu_i and covariance matrices \Sigma_i
    
    The model of a particular word is represented by the model parameters
    \lambda_w = {{a_ij},\mu_i,\Sigma_i}. In state i, it has its own transition (log probability), a mean vector and a covariance matrix
    
    N word models make u the parameter set of the acousitc model \Lambda = {\lambda_w1,...\lambda_wN}
    
    We want to compute the likelyhood of the data given the model.
    P(O|\lambda)
    
    EACH WORD HAS ITS OWN MODEL!
    
    How do we compute P(O|\lambda) ? Remember the rule of complete probability
    
    X represents a path aka its own kind of sequence of state and we have to sum over all possible sequences.
    
    P(O|\labda_w) = sum(X, p(O\wedge X |\lambda_w)) = sum(x, p(O| X \wedge \lambda_w)*P(X|\lambda_w))
    
    Assume conditional independence.
    The likelihood for the joint event p(O\wedge X|\lambda_w) can be given by p(O| X \wedge \lambda_w)*P(X|\lambda_w)
    
    Summing over all possible state sequences  explodes for large number of states.
    
    In practice, given the specific structure of HMM, we are implicitly computing all possible state sequences just not storing them.
    Only store the ones you need in order to rule out the ones which are less important. That is the power of the Viterbi algorithm.
    
    p(O|\lambda_w)=sum(X, P(O\wedge X|\lambda_w)) \approx  P(O\wedge X^{*}|\lambda_w)
    
    
    
    Attributes:
        num_states: total number of HMM states (including two non-emitting states)
        states: a list of HMM states
        log_transp: transition probability matrix (log domain): a[i,j] = log_transp[i,j]
            A is n by n square matrix, a[i,j] is the log probability of state q_i to state q_j
    """

    def __init__(self, num_states=3, num_mixtures=1, self_transp=0.9):
        """Performs GMM evaluation given a list of GMMs and utterances

        Args:
            num_states: number of emitting states
            num_mixture: number of Gaussian mixture components for each state
            self_transp: initial self-transition probability for each state
        """
        self.num_states = num_states
        self.states = [GaussianMixture(n_components=num_mixtures, covariance_type='diag', 
            init_params='kmeans', max_iter=10) for state_id in range(self.num_states)]

        # Initialise transition probability for a left-to-right no-skip HMM
        # For a 3-state HMM, this looks like
        #   [0.9, 0.1, 0. ],
        #   [0. , 0.9, 0.1],
        #   [0. , 0. , 0.9]
        transp = np.diag((1-self_transp)*np.ones(self.num_states-1,),1) + np.diag(self_transp*np.ones(self.num_states,))
        self.log_transp = np.log(transp)


    def viterbi_decoding(self, obs):
        """Performs Viterbi decoding
        
        Remember the Markov assumption
        A path X = [x(1),x(2),...,x(T)] where x(t) is a number (index) for a state
        
        Define the likelihood of a partial path of length t which ends in state j.
        \phi_j(t) = p([o_1,o_2,...,o_t]\wedge[x(1),...,x(t-1),j] \lambda)
        
        The likelyhood of the partial path can be expressed in terms of the likelihood at the preceding state x(t-1)
        \phi_j(t)=a_{x(t-1),j} * b_j(o_t)*\phi_{x(t-1)}(t-1)
        
        Assume the algorithm starts in the final state as this state necessarily belongs to the optimal path X^{*}. Use the equation above
        to express the join likelihood \phi_N(T+1) = P(O\wedge X^{*}|\lambda) in terms of the previous state.
        where N is a state
        \phi_N(T+1)=a_{x(T),N} * \phi_{x(T)}(T)
        
        Initialisation
            For time t=0
            Assign 1 as the likelihood for j=1
            Assign 0 as the likelihood for all other j from 1 up to and including N
            
            For all other t from 1 up to and including T assign 0 as the likeliehood

        Recursion
            Outermost loop is through time t=1 up to and including T
                Innermost loop is over states of the model j=2 to N (i.e. last one is N-1)
                    Compute the likelihood of j at t after finding state k that gives the highest value of \phi_k(t-1)*a_{kj}
                    With k for time t found we have a value for the likelihood for j at time t
                    \phi_j(t)=\phi_k(t-1)*a_{kj}*b_j(o_t)
                    Store the predecessor node pred_k(t) in a matrix (back_pointers!!)
                    
        Termination
            At the end output the probability \phi_k(T)*a_{kN}
            The most likely path can be recovered by tracing back the predecessor information stored at each node pred_k(t)
        Args:
            obs: a sequence of observations [T x dim]
                Matrix O
                dim being the size of the observation vectors, T being greatest value time t can have
            
        Returns:
            log_prob: log-likelihood
            state_seq: most probable state sequence
        """

        # Length of obs sequence
        T = obs.shape[0]

        # Precompute log output probabilities [num_states x T]
        log_outp = np.array([self.states[state_id].score_samples(obs).T for state_id in range(self.num_states)])

        # Initial state probs PI
        initial_dist = np.zeros(self.num_states) # prior prob = log(1) for the first state
        initial_dist[1:] = -float('inf') # prior prob = log(0) for all the other states

        # Back-tracing matrix [num_states x T]
        back_pointers = np.zeros((self.num_states, T), dtype='int')

        # -----------------------------------------------------------------
        # INITIALISATION
        # YOU MAY WANT TO DEFINE THE DELTA VARIABLE AS A MATRIX INSTEAD 
        # OF AN ARRAY FOR AN EASIER IMPLEMENTATION.
        # -----------------------------------------------------------------
        # Initialise the Delta probability
        probs = log_outp[:,0] + initial_dist

        # -----------------------------------------------------------------
        # RECURSION
        # -----------------------------------------------------------------
        for t in range(1, T):
            # ====>>>>
            # ====>>>> FILL WITH YOUR CODE HERE FOLLOWING THE STEPS BELOW.
            # ====>>>>
        
            # STEP 1. Add all transitions to previous best probs

            # STEP 2. Select the previous best state from all transitions into a state

            # STEP 3. Record back-trace information in back_pointers

            # STEP 4. Add output probs to previous best probs


        # -----------------------------------------------------------------
        # SAVE THE GLOBAL LOG LIKELIHOOD IN log_prob AS A RETURN VALUE.
        # THE GLOBAL LOG LIKELIHOOD WILL BE THE VALUE FROM THE LAST STATE 
        # AT TIME T (THE LAST FRAME).
        # -----------------------------------------------------------------
        log_prob = probs[-1]

        # -----------------------------------------------------------------
        # BACK-TRACING: SAVE THE MOST PROBABLE STATE SEQUENCE IN state_seq
        # AS A RETURN VALUE.
        # -----------------------------------------------------------------
        # Allocate state_seq for saving the best state sequence
        state_seq = np.empty((T,), dtype='int')

        # Make sure we finish in the last state
        state_seq[T-1] = self.num_states - 1
        
        # ====>>>>
        # ====>>>> FILL WITH YOUR CODE HERE FOR BACK-TRACING
        # ====>>>>

        # -----------------------------------------------------------------
        # RETURN THE OVERAL LOG LIKELIHOOD log_prob AND THE MOST PROBABLE
        # STATE SEQUENCE state_seq
        # YOU MAY WANT TO CHECK IF THE STATE SEQUENCE LOOKS REASONABLE HERE
        # E.G. FOR A 5-STATE HMM IT SHOULD LOOK SOMETHING LIKE
        #     0 0 0 0 1 1 1 2 2 2 2 3 3 3 3 3 3 3 4 4
        # -----------------------------------------------------------------
        return log_prob, state_seq


