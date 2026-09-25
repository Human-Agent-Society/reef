# Self-Distillation Policy Optimization

SDPO trains the policy against an EMA teacher that reads successful sibling
solutions or environment feedback while scoring the student's original response.
It implements the Section 3 objective from [Reinforcement Learning via
Self-Distillation](https://arxiv.org/abs/2601.20802), using the
[author's pinned implementation](https://github.com/lasgroup/SDPO/tree/7c457fc1b1f636ae794eb0362ba37d4743b06fbc).

- [Recipe and report contract](../../docs/user-guide/recipes/sdpo.rst)
- [Paper reproduction and reference checks](examples/paper/README.md)

The recipe requires the Slime training environment. CPU tests verify the
processor and loss mathematics; a successful numerical check does not establish
paper learning-curve reproduction or qualify a GPU deployment.
