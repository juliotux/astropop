from astropop.pipelines._base import Manager, Config, Instrument, Stage, Product


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

    def callback(instrument, variables, config):
        a = config.get('a')
        b = config.get('b')
        c = config.get('c')
        d = variables.get('d')

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

    def callback(instrument, variables, config):
        c_str = config.get('c_str')
        c = variables.get('a_c_number')
        s = variables.get('dummy_sum')
        m = variablees.get('dummy_mult')

        string_c = "{}{}".format(c_str, c)
        string_abbd = instrument.gen_string(s, m)

        return {'string_c': string_c,
                'string_abbd': string_abbd}

cstring = ''
astring = ''


class GlobalStage(Stage):
    _default_config = dict()
    _required_variables = ['string_c', 'string_abbd']

    def callback(instrument, variables, config):
        cs = variables.get('string_c')
        ad = variables.get('string_abbd')

        global cstring
        global astring

        cstring = cs
        astring = ad

        return {}


class TestManager(Manager):
    def setup_pipeline(self):
        self.register_stage('sum', SumStage(self.factory))
        self.register_stage('string', StringStage(self.factory))
        self.register_stage('global', GlobalStage(self.factory))

    def setup_products(self, a, b, c, d):
        i = DummyInstrument()
        p = Product(manager=self, instrument=i)
        self.factory.set_value(self, 'd', d)
        self.add_product('first_product', p)

m = TestManager()
m.setup_pipeline()
m.setup_products(1, 2, 3, 4)
m.run()
