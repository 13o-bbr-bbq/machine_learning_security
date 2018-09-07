# -*- coding: utf-8 -*-
from util import Utilty
from ga_main import GeneticAlgorithm

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

if __name__ == "__main__":
    util = Utilty()

    # Execute genetic algorithm.
    util.print_message(NOTE, 'Start genetic algorithm.')
    ga = GeneticAlgorithm()
    ga.main()
