import re

class Utils:

    @staticmethod
    def configure_paths(dir_path, relative_path, delimiter, regex):
        lines = []
        with open(dir_path + relative_path, 'r+') as fp:
            lines = fp.readlines()
            for i in range(0, len(lines)):
                line_components = lines[i].split(delimiter)
                if re.match(regex, line_components[-1]):
                    line_components[-1] = dir_path + "/" + line_components[-1]
                    lines[i] = delimiter.join(line_components)
            print(lines)

        with open(dir_path + relative_path, 'w') as fp:
            fp.writelines(lines)