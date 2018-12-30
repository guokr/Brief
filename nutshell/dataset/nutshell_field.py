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


class NutshellSourceField(NutshellBaseField):
    def __init__(self, **kwargs):
        kwargs["eos_token"]  = "<eos>"
        # kwargs["preprocessing"] = lambda seq: seq + [self._eos_token]

        super(NutshellSourceField, self).__init__(**kwargs)


class NutshellTargetField(NutshellBaseField):
    """
    Wrapper of original torchtext data Field
    """
    def __init__(self, **kwargs):
        kwargs["eos_token"] = "<eos>"
        kwargs["init_token"] = "<sos>"
        # kwargs["preprocessing"] = lambda seq: self._sos_token + seq + self._eos_token

        super(NutshellTargetField, self).__init__(**kwargs)


    # def build_vocab(self):


