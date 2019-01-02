#!/usr/bin/env python
# encoding: utf-8

import torchtext


class NutshellBaseField(torchtext.data.Field):
    def __init__(self, **kwargs):
        kwargs["sequential"] = True
        kwargs["tokenize"] = lambda x: x.split()
        kwargs["lower"] = True
        kwargs["batch_first"] = True
        super(NutshellBaseField, self).__init__(**kwargs)

    def build_vocab(self, args):
        super(NutshellBaseField, self).build_vocab(args["Dataset"])


class NutshellSourceField(NutshellBaseField):
    def __init__(self, **kwargs):
        kwargs["eos_token"]  = "<eos>"
        super(NutshellSourceField, self).__init__(**kwargs)


class NutshellTargetField(NutshellBaseField):
    """
    Wrapper of original torchtext data Field
    """
    def __init__(self, **kwargs):
        kwargs["eos_token"] = "<eos>"
        kwargs["init_token"] = "<sos>"

        super(NutshellTargetField, self).__init__(**kwargs)

