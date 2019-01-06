#!/usr/bin/env python
# encoding: utf-8

import torchtext


class BriefBaseField(torchtext.data.Field):
    def __init__(self, **kwargs):
        kwargs["sequential"] = True
        kwargs["tokenize"] = lambda x: x.split()
        kwargs["lower"] = True
        kwargs["batch_first"] = True
        super(BriefBaseField, self).__init__(**kwargs)

    def build_vocab(self, args, **kwargs):
        super(BriefBaseField, self).build_vocab(args["Dataset"], **kwargs)


class BriefSourceField(BriefBaseField):
    def __init__(self, **kwargs):
        kwargs["eos_token"]  = "<eos>"
        super(BriefSourceField, self).__init__(**kwargs)


class BriefTargetField(BriefBaseField):
    """
    Wrapper of original torchtext data Field
    """
    def __init__(self, **kwargs):
        kwargs["eos_token"] = "<eos>"
        kwargs["init_token"] = "<sos>"

        super(BriefTargetField, self).__init__(**kwargs)

