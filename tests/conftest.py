"""Shared test configuration, including offline catalog response playback."""

from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit
from unittest.mock import patch

import pytest
import yaml


class CatalogResponses:
    """Keep HTTP fixture setup separate from catalog assertions."""

    directory = Path(__file__).parent / 'data' / 'catalog_responses'

    @staticmethod
    def load_cassette(cassette_path, serializer):
        from vcr.serialize import deserialize

        requests, responses = [], []
        for path in sorted(Path(cassette_path).glob('*.yaml')):
            saved_requests, saved_responses = deserialize(
                path.read_text(encoding='utf-8'), serializer)
            requests.extend(saved_requests)
            responses.extend(saved_responses)
        return requests, responses

    @staticmethod
    def catalog_name(request, tables):
        host = urlsplit(request.uri).hostname
        if host == 'simbad.cds.unistra.fr':
            return 'simbad'
        if host == 'gea.esac.esa.int':
            return 'gaia'
        if host == 'vizier.cds.unistra.fr':
            body = request.body or ''
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            for line in body.splitlines():
                if line.startswith('-source='):
                    return tables[line.removeprefix('-source=')]
        raise ValueError(f'Unknown catalog response: {request.uri}')

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        from vcr.serialize import serialize

        configs = Path(__file__).parents[1] / 'astropop/catalogs/vizier_catalogs'
        tables = {yaml.safe_load(path.read_text(encoding='utf-8'))['table']:
                  path.stem for path in configs.glob('*.yml')}
        catalogs = {}
        for request, response in zip(cassette_dict['requests'],
                                     cassette_dict['responses']):
            name = CatalogResponses.catalog_name(request, tables)
            data = catalogs.setdefault(name, {'requests': [], 'responses': []})
            data['requests'].append(request)
            data['responses'].append(response)
        directory = Path(cassette_path)
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in catalogs.items():
            path = directory / f'{name}.yaml'
            text = serialize(data, serializer)
            if not path.exists() or path.read_text(encoding='utf-8') != text:
                path.write_text(text, encoding='utf-8')

    @staticmethod
    def uncached_request(original):
        def request(client, *args, **kwargs):
            # VizieR explicitly passes cache=True, overriding cache_conf.
            kwargs['cache'] = False
            return original(client, *args, **kwargs)
        return request

    @staticmethod
    def simbad_body(body):
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        params = parse_qsl(body or '', keep_blank_values=True)
        normalized = []
        for name, value in params:
            if name == 'QUERY':
                prefix, separator, suffix = value.partition(' FROM ')
                columns = prefix.removeprefix('SELECT ').split(', ')
                # Astroquery adds fields from a set, so their SELECT order can
                # change with Python's hash seed. Only normalize simple named
                # columns; preserve expressions, joins and all query criteria.
                if (prefix.startswith('SELECT ') and separator and
                        all(re.fullmatch(r'\w+\."[^"]+"(?: AS "[^"]+")?', col)
                            for col in columns)):
                    value = ('SELECT ' + ', '.join(sorted(columns)) +
                             separator + suffix)
            normalized.append((name, value))
        return urlencode(sorted(normalized))

    @staticmethod
    def match_body(left, right):
        if '/simbad/sim-tap/' in left.uri:
            assert (CatalogResponses.simbad_body(left.body) ==
                    CatalogResponses.simbad_body(right.body))
        else:
            from vcr.matchers import body
            body(left, right)

    @staticmethod
    def serialize(data):
        # Keep the latest response for a request so repeated parameterized
        # tests share examples instead of generating one file per test.
        unique = {}
        for interaction in data['interactions']:
            request = interaction['request']
            body = request['body']
            if '/simbad/sim-tap/' in request['uri']:
                body = CatalogResponses.simbad_body(body)
            key = (request['method'], request['uri'], body)
            unique[key] = interaction
        data['interactions'] = list(unique.values())
        return yaml.safe_dump(data, sort_keys=False)

    @staticmethod
    def deserialize(text):
        return yaml.safe_load(text)

    @staticmethod
    def sanitize_response(response):
        response['headers'] = {k: v for k, v in response['headers'].items()
                               if k.lower() != 'set-cookie'}
        return response

    @staticmethod
    def suppress_gaia_notification(original):
        def get(handler, subcontext, *args, **kwargs):
            if subcontext == 'notification?action=GetNotifications':
                return SimpleNamespace(status=204)
            return original(handler, subcontext, *args, **kwargs)
        return get


def pytest_addoption(parser):
    parser.addoption('--skip-gaia-online', action='store_true',
                     help='Skip live Gaia catalog queries; keep offline Gaia '
                          'tests and deterministic unit tests.')
    parser.addoption('--record-catalogs', action='store_true',
                     help='Refresh catalog HTTP fixtures '
                          '(requires --remote-data=any).')


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'gaia_query: catalog tests that query Gaia in online mode')
    if (config.getoption('--record-catalogs') and
            config.getoption('--remote-data') != 'any'):
        raise pytest.UsageError('--record-catalogs requires --remote-data=any')

    # Astroquery creates its Gaia singleton during import and fetches a status
    # announcement before pytest fixtures can run. Suppress only that unrelated
    # announcement; every catalog request still uses the selected test mode.
    from astroquery.utils.tap.conn.tapconn import TapConn
    get = CatalogResponses.suppress_gaia_notification(TapConn.execute_tapget)
    with patch.object(TapConn, 'execute_tapget', get):
        import astroquery.gaia  # noqa: F401


def pytest_collection_modifyitems(config, items):
    if (config.getoption('--skip-gaia-online') and
            config.getoption('--remote-data') == 'any'):
        skip = pytest.mark.skip(reason='Live Gaia queries disabled by '
                                       '--skip-gaia-online')
        for item in items:
            if item.get_closest_marker('gaia_query'):
                item.add_marker(skip)


@pytest.fixture(scope='module', autouse=True)
def catalog_responses(request):
    if request.node.path.name != 'test_catalogs_online.py':
        yield
        return

    online = request.config.getoption('--remote-data') == 'any'
    recording = request.config.getoption('--record-catalogs')
    if online and not recording:
        yield
        return

    import vcr

    recorder = vcr.VCR()
    recorder.register_serializer('catalogs', CatalogResponses)
    recorder.register_persister(CatalogResponses)
    recorder.register_matcher('catalog_body', CatalogResponses.match_body)
    with recorder.use_cassette(
            str(CatalogResponses.directory),
            serializer='catalogs', record_mode='all' if recording else 'none',
            allow_playback_repeats=True,
            match_on=['method', 'scheme', 'host', 'port', 'path', 'query',
                      'catalog_body'],
            filter_headers=['authorization', 'cookie'],
            before_record_response=CatalogResponses.sanitize_response):
        yield


@pytest.fixture(autouse=True)
def catalog_cache(request, catalog_responses):
    if request.node.path.name != 'test_catalogs_online.py':
        yield
        return

    from astroquery import cache_conf
    from astroquery.query import BaseQuery
    from astroquery.simbad import Simbad

    # A warm developer cache must not hide requests from the recordings or
    # turn a live test into an offline test.
    Simbad.clear_cache()
    uncached = CatalogResponses.uncached_request(BaseQuery._request)
    with (cache_conf.set_temp('cache_active', False),
          patch.object(BaseQuery, '_request', uncached)):
        yield
