# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Get git related information."""

import os


def is_working_tree_clean() -> bool:
    """Check if the git working tree is clean.

    more info can be found here:
    https://stackoverflow.com/questions/3503879/assign-output-of-os-system-to-a-variable-and-prevent-it-from-being-displayed-on
    """
    git_status = os.popen("git status").read()
    iwtc = "nothing to commit, working tree clean" in git_status
    return iwtc


def get_git_describe() -> str:
    """Get the git describe string.

    more info can be found here:
    https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    """
    git_describe = os.popen("git describe --always").read().rstrip()
    return git_describe
