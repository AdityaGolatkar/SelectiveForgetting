import os
import pickle
import random

class Logger(object):
    def __init__(self, index=None, path='logs/', always_save=True):
        if index is None:
            index = '{:06x}'.format(random.getrandbits(6 * 4))
        self.index = index
        self.filename = os.path.join(path, '{}.p'.format(self.index))
        self._dict = {}
        self.logs = []
        self.always_save = always_save

    def __getitem__(self, k):
        return self._dict[k]

    def __setitem__(self,k,v):
        self._dict[k] = v

    @staticmethod
    def load(filename, path='logs/'):
        if not os.path.isfile(filename):
            filename = os.path.join(path, '{}.p'.format(filename))
        if not os.path.isfile(filename):
            raise ValueError("{} is not a valid filename".format(filename))
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def save(self):
        with open(self.filename,'wb') as f:
            pickle.dump(self, f)

    def get(self, _type):
        l = [x for x in self.logs if x['_type'] == _type]
        l = [x['_data'] if '_data' in x else x for x in l]
        # if len(l) == 1:
        #     return l[0]
        return l

    def append(self, _type, *args, **kwargs):
        kwargs['_type'] = _type
        if len(args)==1:
            kwargs['_data'] = args[0]
        elif len(args) > 1:
            kwargs['_data'] = args
        self.logs.append(kwargs)
        if self.always_save:
            self.save()