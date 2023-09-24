'''
Created on Sep 18, 2023

@author: amandapotts
'''
from configparser import ConfigParser


class ModelConfig(ConfigParser):
    '''
    classdocs
    '''

    def __init__(self, filename):
        '''
        Constructor
        '''

        super().__init__(self)
        self.read(filename)

    def output(self, filename):
        with open(filename, 'w') as configfile:
            self.write(configfile, space_around_delimiters=' ')
            
    def write_config_value(self, section, key, value):
        if(section not in self.keys()):
            self[section] = {}
        self.set(section, key, value)


if __name__ == "__main__":
    pass
    
