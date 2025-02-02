import yaml

class ParseConf(dict):
    def __init__(self, conf_path):
        if yaml.__version__ < '5.1.0':
            with open(conf_path, mode='r', encoding='utf-8') as f:
                yml_conf = yaml.load(f)
        else:
            with open(conf_path, mode='r', encoding='utf-8') as f:
                yml_conf = yaml.load(f, Loader=yaml.FullLoader)

        super().update(yml_conf)

    def __getitem__(self, key):
        current = self
        for k in key.split('.'):
            current = dict.__getitem__(current, k)
        return current

    def __setitem__(self, key, value):
        keys = key.split('.')
        current = self
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        dict.__setitem__(current, keys[-1], value)

    def get(self, keys, default=None):
        try:
            return self[keys]
        except KeyError:
            return default