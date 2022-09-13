from .dynamic_oracle import DynamicOracle
from parsebrain.processing.dependency_parsing.transition_based.configuration import GoldConfiguration, Word
from parsebrain.processing.dependency_parsing.transition_based.transition import ArcEagerTransition
"""
Need to think how to cleanly deal 
"""


# Dynamic oracle from "A Dynamic Oracle for Arc-Eager Dependency Parsing" Goldberg, Yoav, and Joakim Nivre

# Maybe remove the doctest and use pytest here.
# Maybe need to rework the actions the parser can take in a specific config. ( to not have multiple implementation of
# those rules)

class DynamicOracleArcEager(DynamicOracle):
    def get_oracle_move_from_config_tree(self, configuration, gold_configuration):
        '''
        Compute the next oracle move from current configuration and the gold configuration

        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import Configuration
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import ArcEagerTransition
        >>> config = Configuration()
        >>> config.buffer_string = [Word("j",1),Word("ai",2),Word("vu",3),
        ... Word("jamais",4), Word("eu",5),Word("ça",6),Word("pour",7),
        ... Word("un",8),Word("une",9),Word("un",10),Word("colloque",11)]
        >>> config.buffer = ["j","ai","vu","jamais","eu","ça","pour","un","une","un","colloque"] # will be embedding
        >>> config_gold = GoldConfiguration()
        >>> heads = [5, 3, 5, 5, 0, 5, 5, 11, 8, 9, 7]
        >>> for i, h in enumerate(heads): config_gold.heads[i+1] = h
        >>> deps = [[], [], [2], [], [1,3,4,6,7], [], [11], [9], [10], [], [8]]
        >>> for i, d in enumerate(deps): config_gold.deps[i+1] = d
        >>> oracle = DynamicOracleArcEager()
        >>> m = oracle.get_oracle_move_from_config_tree(config, config_gold)
        >>> print(m)
        0
        >>> transiton = ArcEagerTransition()
        >>> list_decision = []
        >>> while len(config.buffer) != 0:
        ...     decision = oracle.get_oracle_move_from_config_tree(config, config_gold)
        ...     list_decision.append(decision)
        ...     config = transiton._apply_decision(decision, config)
        >>> print(list_decision)
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 2, 3, 2, 0, 2, 2, 3, 3, 1, 2]
        >>>

        '''
        #print(f"{[str(w) for w in configuration.buffer_string]}")
        if len(configuration.buffer) == 0: #if config is terminal with Arc eager:
            return -1 # padding
        decision_cost = [self.compute_shift_cost(configuration, gold_configuration)
                         + int(not ArcEagerTransition.shift_condition(configuration)) * 99999,
                         self.compute_left_arc_cost(configuration, gold_configuration)
                         +int(not ArcEagerTransition.left_arc_condition(configuration)) * 99999,
                         self.compute_right_arc_cost(configuration, gold_configuration)
                         +int(not ArcEagerTransition.right_arc_condition(configuration)) * 99999,
                         self.compute_reduce_cost(configuration, gold_configuration)
                         +int(not ArcEagerTransition.reduce_condition(configuration)) * 99999]
        min_cost = min(decision_cost)
        return decision_cost.index(min_cost)



    def compute_shift_cost(self, configuration, gold_configuration):
        '''
        (SHIFT; c, Ggold): Pushing b onto the stack means that b will not be able to acquire any
        head or dependents in s|σ. The cost is therefore the number of arcs in Agold of the form
        (k, l′, b) or (b, l′, k) such that k ∈ s|σ.
        (SH; c = (σ, b|β, A), Gg) = {(k, l′, b) ∈ Ag |k ∈ σ} ∪ {(b, l′, k) ∈ Ag |k ∈ σ}

        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import Configuration
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import ArcEagerTransition
        >>> config = Configuration()
        >>> config.buffer_string = [Word("j",1),Word("ai",2),Word("vu",3),
        ... Word("jamais",4),Word("eu",5),Word("ça",6),Word("pour",7),
        ... Word("un",8),Word("une",9),Word("un",10),Word("colloque",11)]
        >>> config.buffer = ["j","ai","vu","jamais","eu","ça","pour","une","une","un","colloque"] # will be embedding
        >>> config_gold = GoldConfiguration()
        >>> heads = [5, 3, 5, 5, 0, 5, 5, 11, 8, 9, 7]
        >>> for i, h in enumerate(heads): config_gold.heads[i+1] = h
        >>> deps = [[], [], [2], [], [1,3,4,6,7], [], [11], [9], [10], [], [8]]
        >>> for i, d in enumerate(deps): config_gold.deps[i+1] = d
        >>> oracle = DynamicOracleArcEager()
        >>> c = oracle.compute_shift_cost(config, config_gold)
        >>> print(c)
        0
        >>> transition = ArcEagerTransition()
        >>> config = transition.shift(config)
        >>> config = transition.shift(config)
        >>> c = oracle.compute_shift_cost(config, config_gold)
        >>> print(c)
        1
        >>>

        '''

        # nothing to shift invalid command.
        if len(configuration.buffer_string) <= 1:
            return 99999
        cost = 0
        b = configuration.buffer_string[0].position
        for s_e in configuration.stack_string:
            p = s_e.position
            if gold_configuration.heads[p] == b or (gold_configuration.heads[b] == p):
                cost += 1
        return cost

    def compute_reduce_cost(self, configuration, gold_configuration):
        '''
        (REDUCE; c, Ggold): Popping s from the stack means that s will not be able to acquire
        any dependents in b|β. The cost is therefore the number of arcs in Agold of the form
        (s, l′, k) such that k ∈ b|β. While it may seem that a gold arc of the form (k, l, s) should be
        accounted for as well, note that a gold arc of that form, if it exists, is already accounted
        for by a previous (erroneous) RIGHT-ARCl transition when s acquired its head.
        (RE; c = (σ|s, β, A), Gg) = {(s, l′, k) ∈ Ag |k ∈ β}

        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import Configuration
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import ArcEagerTransition
        >>> config = Configuration()
        >>> config.buffer_string = [Word("j",1),Word("ai",2),Word("vu",3),
        ... Word("jamais",4),Word("eu",5),Word("ça",6),Word("pour",7),
        ... Word("un",8),Word("une",9),Word("un",10),Word("colloque",11)]
        >>> config.buffer = ["j","ai","vu","jamais","eu","ça","pour","une","une","un","colloque"] # will be embedding
        >>> config_gold = GoldConfiguration()
        >>> heads = [5, 3, 5, 5, 0, 5, 5, 11, 8, 9, 7]
        >>> for i, h in enumerate(heads): config_gold.heads[i+1] = h
        >>> deps = [[], [], [2], [], [1,3,4,6,7], [], [11], [9], [10], [], [8]]
        >>> for i, d in enumerate(deps): config_gold.deps[i+1] = d
        >>> oracle = DynamicOracleArcEager()
        >>> c = oracle.compute_reduce_cost(config, config_gold)
        >>> print(c)
        99999
        >>> transition = ArcEagerTransition()
        >>> config = transition.shift(config)
        >>> c = oracle.compute_reduce_cost(config, config_gold)
        >>> print(c)
        0
        >>> config = transition.shift(config)
        >>> config = transition.shift(config)
        >>> config = transition.shift(config)
        >>> config = transition.shift(config)
        >>> c = oracle.compute_reduce_cost(config, config_gold)
        >>> print(c)
        2
        >>>
        '''
        if len(configuration.stack_string) == 0:
            return 99999
        cost = 0
        s = configuration.stack_string[-1].position
        for b_e in configuration.buffer_string:
            p = b_e.position
            if gold_configuration.heads[p] == s:
                cost += 1
        return cost

    def compute_left_arc_cost(self, configuration, gold_configuration):
        '''
        (LEFT-ARCl ; c, Ggold): Adding the arc (b, l, s) and popping s from the stack means that s
        will not be able to acquire any head or dependents in β. The cost is therefore the number
        of arcs in Agold of the form (k, l′, s) or (s, l′, k) such that k ∈ β. Note that the cost is 0 for
        the trivial case where (b, l, s) ∈ Agold, but also for the case where b is not the gold head of
        s but the real head is not in β (due to an erroneous previous transition) and there are no
        gold dependents of s in β.6
        (LAl ; c = (σ|s, b|β, A), Gg) = {(k, l′, s) ∈ Ag |k ∈ β} ∪ {(s, l′, k) ∈ Ag |k ∈ β}
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import Configuration
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import ArcEagerTransition
        >>> config = Configuration()
        >>> config.buffer_string = [Word("j",1),Word("ai",2),Word("vu",3),
        ... Word("jamais",4),Word("eu",5),Word("ça",6),Word("pour",7),
        ... Word("un",8),Word("une",9),Word("un",10),Word("colloque",11)]
        >>> config.buffer = ["j","ai","vu","jamais","eu","ça","pour","une","une","un","colloque"] # will be embedding
        >>> config_gold = GoldConfiguration()
        >>> heads = [5, 3, 5, 5, 0, 5, 5, 11, 8, 9, 7]
        >>> for i, h in enumerate(heads): config_gold.heads[i+1] = h
        >>> deps = [[], [], [2], [], [1, 3, 4, 6, 7], [], [11], [9], [10], [], [8]]
        >>> for i, d in enumerate(deps): config_gold.deps[i+1] = d
        >>> oracle = DynamicOracleArcEager()
        >>> c = oracle.compute_left_arc_cost(config, config_gold)
        >>> print(c)
        99999
        >>> transition = ArcEagerTransition()
        >>> config = transition.shift(config)
        >>> c = oracle.compute_left_arc_cost(config, config_gold)
        >>> print(c)
        1
        >>> config = transition.shift(config)
        >>> config = transition.shift(config)
        >>> c = oracle.compute_left_arc_cost(config, config_gold)
        >>> print(c)
        2

        '''
        if len(configuration.stack_string) == 0:
            return 99999
        cost = 0
        s = configuration.stack_string[-1].position
        # checking in the buffer (is the head the right one and is there any right dependent to this element ?)
        for b_e in configuration.buffer_string:
            p = b_e.position
            if gold_configuration.heads[p] == s or gold_configuration.heads[s] == p:
                # if transition is correct, first element of buffer is the head (no cost)
                if gold_configuration.heads[s] == p and configuration.buffer_string[0].position == p:
                    continue
                cost += 1
        # checking in the stack ( is there left dependent ?)
        for s_e in configuration.stack_string:
            p = s_e.position
            if gold_configuration.heads[p] == s or gold_configuration.heads[s] == p:
                cost += 1
        return cost

    def compute_right_arc_cost(self, configuration, gold_configuration):
        '''
        RIGHT-ARCl ; c, Ggold): Adding the arc (s, l, b) and pushing b onto the stack means that
        b will not be able to acquire any head in σ or β, nor any dependents in σ. The cost is
        therefore the number of arcs in Agold of the form (k, l′, b), such that k ∈ σ or k ∈ β, or of
        the form (b, l′, k) such that k ∈ σ. Note again that the cost is 0 for the trivial case where
        (s, l, b) ∈ Agold, but also for the case where s is not the gold head of b but the real head is
        not in σ or β (due to an erroneous previous transition) and there are no gold dependents
        of b in σ.
        (RAl ; c = (σ|s, b|β, A), Gg) = {(k, l′, b) ∈ Ag |k ∈ σ ∨ k ∈ β} ∪ {(b, l′, k) ∈ Ag |k ∈ σ}
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import Configuration
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import ArcEagerTransition
        >>> config = Configuration()
        >>> config.buffer_string = [Word("j",1),Word("ai",2),Word("vu",3),
        ... Word("jamais",4),Word("eu",5),Word("ça",6),Word("pour",7),
        ... Word("un",8),Word("une",9),Word("un",10),Word("colloque",11)]
        >>> config.buffer = ["j","ai","vu","jamais","eu","ça","pour","une","une","un","colloque"] # will be embedding
        >>> config_gold = GoldConfiguration()
        >>> heads = [5, 3, 5, 5, 0, 5, 5, 11, 8, 9, 7]
        >>> for i, h in enumerate(heads): config_gold.heads[i+1] = h
        >>> deps = [[], [], [2], [], [1,3,4,6,7], [], [11], [9], [10], [], [8]]
        >>> for i, d in enumerate(deps): config_gold.deps[i+1] = d
        >>> oracle = DynamicOracleArcEager()
        >>> transition = ArcEagerTransition()
        >>> config = transition.shift(config)
        >>> c = oracle.compute_right_arc_cost(config, config_gold)
        >>> print(c)
        1
        >>> config = transition.shift(config)
        >>> c = oracle.compute_right_arc_cost(config, config_gold)
        >>> print(c)
        2
        >>>


        '''

        if len(configuration.buffer_string) == 0 or len(configuration.stack_string) == 0:
            return 99999
        cost = 0
        b = configuration.buffer_string[0].position
        for b_e in configuration.buffer_string:
            p = b_e.position
            if gold_configuration.heads[b] == p:
                cost += 1
        for s_e in configuration.stack_string:
            p = s_e.position
            # if the head was deeper in the stack or if there is dependent in the stack.
            if gold_configuration.heads[b] == p or gold_configuration.heads[p] == b:
                # exception for the last element of the stack (should be the head)
                if gold_configuration.heads[b] == p and s_e.position == configuration.stack_string[-1].position:
                    continue
                cost += 1

        return cost

if __name__ == "__main__":
    import doctest

    doctest.testmod()
