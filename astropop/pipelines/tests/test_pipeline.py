from astropop.pipelines import Manager, Config, Instrument, Stage, Product

# from astropop.logger import logger
# logger.setLevel('DEBUG')


# TODO: implement unittest https://docs.python.org/3/library/unittest.html
# TODO: set values to global variables and assert them

string_store = []

class DummyInstrument(Instrument):
    a = 'a+b='
    b = 'b*d='
    _identifier = 'test.instrument'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sum_numbers(self, a, b):
        return a+b

    def multiply_numbers(self, b, d):
        return b*d

    def gen_string(self, ab, bd):
        return "{}{} {}{}".format(self.a, ab,
                                  self.b, bd)


class SumStage(Stage):
    _default_config = dict(a=2, b=3, c=1)
    _required_variables = ['d']
    _requested_functions = ['sum_numbers', 'multiply_numbers']
    _provided = ['dummy_sum', 'a_c_number', 'dummy_mult']
    _raise_error = True

    @staticmethod
    def callback(instrument, variables, config, logger):
        a = config.get('a')
        b = config.get('b')
        c = config.get('c')
        d = variables.get('d')

        logger.warn('Testing Warnings')

        s = instrument.sum_numbers(a, b)
        m = instrument.multiply_numbers(b, d)

        return {'dummy_sum': s,
                'a_c_number': c,
                'dummy_mult': m}


class StringStage(Stage):
    _default_config = dict(c_str='c=')
    _required_variables = ['dummy_sum', 'a_c_number']
    _optional_variables = ['dummy_mult']
    _requested_functions = ['gen_string']
    _provided = ['string_c', 'string_abbd']
    _raise_error = True

    @staticmethod
    def callback(instrument, variables, config, logger):
        logger.debug(config)
        c_str = config.get('c_str')
        c = variables.get('a_c_number')
        s = variables.get('dummy_sum')
        m = variables.get('dummy_mult')

        string_c = "{}{}".format(c_str, c)
        string_abbd = instrument.gen_string(s, m)

        return {'string_c': string_c,
                'string_abbd': string_abbd}


class GlobalStage(Stage):
    _default_config = dict()
    _required_variables = ['string_c', 'string_abbd']
    _raise_error = True

    @staticmethod
    def callback(instrument, variables, config, logger):
        cs = variables.get('string_c')
        ad = variables.get('string_abbd')

        global string_store
        string_store.append((cs, ad))

        return {}


class TestManager(Manager):
    config = Config(stages={'sum': {'a': 1, 'b': 2, 'c': 3}})

    def setup_pipeline(self):
        self.register_stage('sum', SumStage(self.factory))
        self.register_stage('string', StringStage(self.factory))
        self.register_stage('globalvars', GlobalStage(self.factory))

    def setup_products(self, name, d):
        i = DummyInstrument()
        p = Product(manager=self, instrument=i)
        self.add_product(name, p)
        p.add_target('globalvars')
        self.set_value(p, 'd', d)


def test_pipeline():
    m = TestManager()
    m.setup_pipeline()
    m.setup_products('first_product', 4)
    m.setup_products('secon_product', 8)
    m.show_products()
    m.run()
    m.run(index=0, target='sum')

    # TODO: Include the tests here
    assert string_store == [('c=3', 'a+b=3 b*d=16'), ('c=3', 'a+b=3 b*d=16')]


if __name__ == '__main__':
    test_pipeline()
