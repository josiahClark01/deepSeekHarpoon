DeepSeekHarpoon
===============

(WIP) This repo is an attempt at recreating the deepseek v3 pretraining model for academic and research purposes.

- currently only uses torch so no fancy parallelism yet outside of native pytorch bits, goal is to keep it simple and explainable.

.. note:: We used the name harpoon because pytorch OOTB parallelism scheme pushed us to make the model long and skinny from a parrallelism perspective, hence the name harpoon... Arghhhhh!!!!!!!

Prerequisits 
------------

- HF_Token account and token to access ds tokenizer
- cluster or node to run this on.
- pytorch environment w/ all the trimmings.
- fineweb training dataset (pulled for you from HF)


Repo Contents
-------------

This repo contains: 

- `Model definition <model.py>`_: fairly generic layout.
- `Main training loop and some helper utils <train.py>`_: some knobs for adjusting model arch based on classic transformer params (fiddle with these to make model larger or smaller), pulls the dataset, and tokenizer, and trains the model.
- `CSV w/ back the napkin GEMM calcs <dsv3GEMMCalcs.csv>`_: GEMM dimension calcs, mostly back of the napkin for the curious at heart or someone interested in doing something hardcore..
- `Course generic megatron parallelism calculator <model.py>`_: basic tool for precalculating if a parallelism scheme will fit into memory using megatron, this repo doesn't currently use megatron, but was considering it before doing the PT implimentation so someone might find this calculator handy, likely not given the rate at which megatron changes.
- Other typical repo bits, but above is the primary stuff.
