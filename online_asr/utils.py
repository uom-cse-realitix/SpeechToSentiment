import yaml


class Utils:

    @staticmethod
    def readYaml(configFileName):
        config = {}
        with open(configFileName, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config