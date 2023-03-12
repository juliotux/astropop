# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage SQL databases in a simplier way."""

import sqlite3 as sql
import numpy as np
import base64
from astropy.table import Table

from .logger import logger
from .py_utils import broadcast


__all__ = ['SQLDatabase', 'SQLTable', 'SQLRow', 'SQLColumn']


_ID_KEY = '__id__'
_B32_PREFIX = '__b32c__'


def _b32_encode(data):
    """Encode the data to base32."""
    # encode and remove padding
    return base64.b32encode(data.encode()).decode().strip('=')


def _b32_decode(data):
    """Decode the data from base32."""
    data = data.upper()
    # check padding
    if len(data) % 8 != 0:
        data += '='*(8 - len(data) % 8)
    return base64.b32decode(data.encode()).decode()


def _sanitize_colnames(data, allow_b32_encode=False):
    """Sanitize the colnames to avoid invalid characteres like '-'."""
    def _sanitize(key):
        if len([ch for ch in key if not ch.isalnum() and ch != '_']) != 0:
            if not allow_b32_encode:
                raise ValueError(f'Invalid column name: {key}.')
            key = f"{_B32_PREFIX}{_b32_encode(key)}"
        return key.lower()

    if isinstance(data, dict):
        d = data
        colnames = _sanitize_colnames(list(data.keys()))
        return dict(zip(colnames, d.values()))
    if isinstance(data, str):
        return _sanitize(data)
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f'{type(data)} is not supported.')

    return [_sanitize(i) for i in data]


def _sanitize_value(data):
    """Sanitize the value to avoid sql errors."""
    if data is None or isinstance(data, bytes):
        return data
    if isinstance(data, (str, np.str_)):
        return f"{data}"
    if np.isscalar(data) and np.isreal(data):
        if isinstance(data, (int, np.integer)):
            return int(data)
        elif isinstance(data, (float, np.floating)):
            return float(data)
    if isinstance(data, (bool, np.bool_)):
        return bool(data)
    raise TypeError(f'{type(data)} is not supported.')


def _fix_row_index(row, length):
    """Fix the row number to be a valid index."""
    if row < 0:
        row += length
    if row >= length or row < 0:
        raise IndexError('Row index out of range.')
    return row


def _dict2row(cols, **row):
    values = [None]*len(cols)
    for i, c in enumerate(cols):
        if c in row.keys():
            values[i] = row[c]
        else:
            values[i] = None
    return values


def _parse_where(where):
    args = None
    if where is None:
        _where = None
    elif isinstance(where, dict):
        where = _sanitize_colnames(where)
        for i, (k, v) in enumerate(where.items()):
            v = _sanitize_value(v)
            if i == 0:
                _where = f"{k}=?"
                args = [v]
            else:
                _where += f" AND {k}=?"
                args.append(v)
    elif isinstance(where, str):
        _where = where
    elif isinstance(where, (list, tuple)):
        for w in where:
            if not isinstance(w, str):
                raise TypeError('if where is a list, it must be a list '
                                f'of strings. Not {type(w)}.')
        _where = ' AND '.join(where)
    else:
        raise TypeError('where must be a string, list of strings or'
                        ' dict.')
    return _where, args


class _SQLViewerBase:
    """Memview for SQL data. Not allowed to copy."""

    def __copy__(self):
        raise NotImplementedError('Cannot copy SQL viewing classes.')

    def __deepcopy__(self, memo):
        raise NotImplementedError('Cannot copy SQL viewing classes.')


class _SQLRowIndexer(_SQLViewerBase):
    """A class for indexing SQL rows. Safer method while removing rows.

    Index is obtained by indexof(self).

    Parameters
    ----------
    row_list: list
        The list of `_SQLRowIndexer` to get the index of.
    """

    def __init__(self, row_list):
        self._row_list = row_list

    @property
    def index(self):
        """Get the index of the row."""
        return self._row_list.index(self)


class _SQLColumnCacher(_SQLViewerBase):
    """Cache and process column names."""

    def __init__(self, db, table, allow_b32_encode=False):
        self._db = db
        self._table = table
        self._columns = {}
        self._allow_encode = allow_b32_encode
        self.rebuild_cache()

    def get_column(self, name):
        """Get a column from the cache."""
        name = self._check_column(name)
        return self._columns[name]

    def add_column(self, name):
        """Add a column to the cache."""
        name = name.lower()
        value = _sanitize_colnames(name, self._allow_encode)
        if self._allow_encode and name.startswith(_B32_PREFIX):
            name = _b32_decode(name[len(_B32_PREFIX):])
        if name in self._columns.keys():
            raise KeyError('Column name already exists: {}'.format(name))
        self._columns[name.lower()] = value

    def remove_column(self, name):
        """Remove a column from the cache."""
        name = self._check_column(name)
        if name not in self._columns.keys():
            raise KeyError('Column name does not exist: {}'.format(name))
        self._columns.pop(name)

    def rebuild_cache(self):
        """Rebuild the cache."""
        self._columns = {}
        for name in self._db._query_colnames(self._table):
            name = name.lower()
            self.add_column(name)

    def columns(self):
        return list(self._columns.keys())

    def _check_column(self, name):
        """Check if the column exists."""
        name = name.lower()
        if name not in self._columns.keys():
            raise KeyError('Column name does not exist: {}'.format(name))
        return name


class SQLTable(_SQLViewerBase):
    """Handle an SQL table operations interfacing with the DB."""

    def __init__(self, db, name):
        """Initialize the table.

        Parameters
        ----------
        db : SQLDatabase
            The parent database object.
        name : str
            The name of the table in the database.
        """
        self._db = db
        self._name = name

    @property
    def name(self):
        """Get the name of the table."""
        return self._name

    @property
    def db(self):
        """Get the database name."""
        return self._db._db

    @property
    def column_names(self):
        """Get the column names of the current table."""
        names = self._db.column_names(self._name)
        return names

    @property
    def values(self):
        """Get the values of the current table."""
        return self.select()

    def select(self, **kwargs):
        """Select rows from the table."""
        where = kwargs.pop('where', None)
        order = kwargs.pop('order', None)
        return self._db.select(self._name, where=where, order=order, **kwargs)

    def as_table(self):
        """Return the current table as an `~astropy.table.Table` object."""
        if len(self) == 0:
            return Table(names=self.column_names)
        return Table(rows=self.values,
                     names=self.column_names)

    def add_column(self, name, data=None):
        """Add a column to the table."""
        self._db.add_column(self._name, name, data=data)

    def add_rows(self, data, add_columns=False):
        """Add a row to the table."""
        self._db.add_rows(self._name, data, add_columns=add_columns)

    def get_column(self, column):
        """Get a given column from the table."""
        return self._db.get_column(self._name, column)

    def get_row(self, row):
        """Get a given row from the table."""
        return self._db.get_row(self._name, row)

    def set_column(self, column, data):
        """Set a given column in the table."""
        self._db.set_column(self._name, column, data)

    def set_row(self, row, data):
        """Set a given row in the table."""
        self._db.set_row(self._name, row, data)

    def delete_column(self, column):
        """Delete a given column from the table."""
        self._db.delete_column(self._name, column)

    def delete_row(self, row):
        """Delete all rows from the table."""
        self._db.delete_row(self._name, row)

    def index_of(self, where):
        """Get the index of the rows that match the given condition."""
        return self._db.index_of(self._name, where)

    def _resolve_tuple(self, key):
        """Resolve how tuples keys are handled."""
        col, row = key
        _tuple_err = """Tuple items must be in the format table[col, row] or
        table[row, col].
        """

        if not isinstance(col, str):
            # Try inverting
            col, row = row, col

        if not isinstance(col, str):
            raise KeyError(_tuple_err)

        if not isinstance(row, (int, slice, list, np.ndarray)):
            raise KeyError(_tuple_err)

        return col, row

    def __getitem__(self, key):
        """Get a row or a column from the table."""
        if isinstance(key, (int, np.int_)):
            return self.get_row(key)
        if isinstance(key, (str, np.str_)):
            return self.get_column(key)
        if isinstance(key, tuple):
            if len(key) not in (1, 2):
                raise KeyError(f'{key}')
            if len(key) == 1:
                return self[key[0]]
            col, row = self._resolve_tuple(key)
            return self[col][row]
        raise KeyError(f'{key}')

    def __setitem__(self, key, value):
        """Set a row or a column in the table."""
        if isinstance(key, int):
            self.set_row(key, value)
        elif isinstance(key, str):
            self.set_column(key, value)
        elif isinstance(key, tuple):
            if len(key) not in (1, 2):
                raise KeyError(f'{key}')
            if len(key) == 1:
                self[key[0]] = value
            else:
                col, row = self._resolve_tuple(key)
                self[col][row] = value
        else:
            raise KeyError(f'{key}')

    def __len__(self):
        """Get the number of rows in the table."""
        return self._db.count(self._name)

    def __contains__(self, item):
        """Check if a given column is in the table."""
        return item in self.column_names

    def __iter__(self):
        """Iterate over the rows of the table."""
        for i in self.select():
            yield i

    def __repr__(self):
        """Get a string representation of the table."""
        s = f"{self.__class__.__name__} '{self.name}'"
        s += f" in database '{self.db}':"
        s += f"({len(self.column_names)} columns x {len(self)} rows)\n"
        s += '\n'.join(self.as_table().__repr__().split('\n')[1:])
        return s


class SQLColumn(_SQLViewerBase):
    """Handle an SQL column operations interfacing with the DB."""

    def __init__(self, db, table, name):
        """Initialize the column.

        Parameters
        ----------
        db : SQLDatabase
            The parent database object.
        table : str
            The name of the table in the database.
        name : str
            The column name in the table.
        """
        self._db = db
        self._table = table
        self._name = name

    @property
    def name(self):
        """Get the name of the column."""
        return self._name

    @property
    def values(self):
        """Get the values of the current column."""
        vals = self._db.select(self._table, columns=[self._name])
        return [i[0] for i in vals]

    @property
    def table(self):
        """Get the table name."""
        return self._table

    def __getitem__(self, key):
        """Get a row from the column."""
        if isinstance(key, (int, np.int_, slice)):
            return self.values[key]
        if isinstance(key, (list, np.ndarray)):
            v = self.values
            return [v[i] for i in key]
        raise IndexError(f'{key}')

    def __setitem__(self, key, value):
        """Set a row in the column."""
        if isinstance(key, (int, np.int_)):
            self._db.set_item(self._table, self._name, key, value)
        elif isinstance(key, (slice, list, np.ndarray)):
            v = np.array(self.values)
            v[key] = value
            self._db.set_column(self._table, self._name, v)
        else:
            raise IndexError(f'{key}')

    def __len__(self):
        """Get the number of rows in the column."""
        return len(self.values)

    def __iter__(self):
        """Iterate over the column."""
        for i in self.values:
            yield i

    def __contains__(self, item):
        """Check if the column contains a given value."""
        return item in self.values

    def __repr__(self):
        """Get a string representation of the column."""
        s = f"{self.__class__.__name__} {self._name} in table '{self._table}'"
        s += f" ({len(self)} rows)"
        return s


class SQLRow(_SQLViewerBase):
    """Handle and SQL table row interfacing with the DB."""

    def __init__(self, db, table, row_indexer):
        """Initialize the row.

        Parameters
        ----------
        db : SQLDatabase
            The parent database object.
        table : str
            The name of the table in the database.
        row_indexer : `~astropop._db._SQLRowIndexer`
            The row index in the table.
        """
        self._db = db
        self._table = table
        self._row_indexer = row_indexer

    @property
    def column_names(self):
        """Get the column names of the current table."""
        names = self._db.column_names(self._table)
        return names

    @property
    def table(self):
        """Get the table name."""
        return self._table

    @property
    def values(self):
        """Get the values of the current row."""
        return self._db.select(self._table, where={_ID_KEY: self.index+1})[0]

    @property
    def index(self):
        """Get the index of the current row."""
        return self._row_indexer.index

    @property
    def keys(self):
        """Get the keys of the current row."""
        return self.column_names

    @property
    def items(self):
        """Get the items of the current row."""
        return zip(self.column_names, self.values)

    def as_dict(self):
        """Get the row as a dict."""
        return dict(self.items)

    def __getitem__(self, key):
        """Get a column from the row."""
        if isinstance(key, (str, np.str_)):
            column = key
            try:
                return self._db.get_item(self._table, column, self.index)
            except ValueError:
                raise KeyError(f'{key}')
        if isinstance(key, (int, np.int_)):
            return self.values[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key, value):
        """Set a column in the row."""
        if not isinstance(key, (str, np.str_)):
            raise KeyError(f'{key}')

        column = key = key.lower()
        if key not in self.column_names:
            raise KeyError(f'{key}')
        self._db.set_item(self._table, column, self.index, value)

    def __iter__(self):
        """Iterate over the row."""
        for i in self.values:
            yield i

    def __contains__(self, item):
        """Check if the row contains a given value."""
        return item in self.values

    def __repr__(self):
        """Get a string representation of the row."""
        s = f"{self.__class__.__name__} {self.index} in table '{self._table}' "
        s += self.as_dict().__repr__()
        return s


class SQLDatabase:
    """Database creation and manipulation with SQL.

    Parameters
    ----------
    db : str
        The name of the database file. If ':memory:' is given, the
        database will be created in memory.
    autocommit : bool (optional)
        Whether to commit changes to the database after each operation.
        Defaults to True.
    allow_colname_encode : bool (optional)
        If True, column names containing not allowed characters will be encoded
        to base36 valid names. The column name will be prefixed with
        '__b36c__' caracters. Defaults to True.

    Notes
    -----
    - '__id__' is only for internal indexing. It is ignored on returns.
    - Any column name with '__b36c__' prefix is expected to be encoded.
    - The column names are case insensitive.
    """

    def __init__(self, db=':memory:', autocommit=True,
                 allow_colname_encode=False):
        # Construct sqlite connection.
        self._db = db
        self._con = sql.connect(self._db)
        self._cur = self._con.cursor()
        self.autocommit = autocommit
        self._allow_encode = allow_colname_encode

        # Use row indexers to preserve row viewing after table changes.
        self._row_indexes = {}
        self._build_row_indexes()

        # Cache the column names for each table. Speeds up queries.
        self._column_cache = {}
        self._build_column_cache()

    def execute(self, command, arguments=None):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))
        try:
            if arguments is None:
                self._cur.execute(command)
            else:
                self._cur.execute(command, arguments)
            res = self._cur.fetchall()
        except sql.Error as e:
            self._con.rollback()
            raise e

        if self.autocommit:
            self.commit()
        return res

    def executemany(self, command, arguments):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))

        try:
            self._cur.executemany(command, arguments)
            res = self._cur.fetchall()
        except sql.Error as e:
            self._con.rollback()
            raise e

        if self.autocommit:
            self.commit()
        return res

    def commit(self):
        """Commit the current transaction."""
        self._con.commit()

    def count(self, table, where=None):
        """Get the number of rows in the table."""
        self._check_table(table)
        comm = "SELECT COUNT(*) FROM "
        comm += f"{table} "
        where, args = _parse_where(where)
        if where is not None:
            comm += f"WHERE {where}"
        comm += ";"
        return self.execute(comm, args)[0][0]

    def select(self, table, columns=None, where=None, order=None, limit=None,
               offset=None):
        """Select rows from a table.

        Parameters
        ----------
        columns : list (optional)
            List of columns to select. If None, select all columns.
        where : dict (optional)
            Dictionary of conditions to select rows. Keys are column names,
            values are values to compare. All rows equal to the values will
            be selected. If None, all rows are selected.
        order : str (optional)
            Column name to order by.
        limit : int (optional)
            Number of rows to select.
        """
        self._check_table(table)
        if columns is None:
            columns = self[table].column_names
        elif isinstance(columns, str):
            columns = [columns]
        # only use sanitized column names
        columns = ', '.join(_sanitize_colnames(columns))

        comm = f"SELECT {columns} "
        comm += f"FROM {table} "
        args = []

        where, args_w = _parse_where(where)
        if where is not None:
            comm += f"WHERE {where} "
            if args_w is not None:
                args += args_w

        if order is not None:
            order = _sanitize_colnames(order)
            comm += f"ORDER BY {order} ASC "

        if limit is not None:
            comm += "LIMIT ? "
            if not isinstance(limit, (int, np.integer)):
                raise TypeError('limit must be an integer.')
            args.append(int(limit))
        if offset is not None:
            if limit is None:
                raise ValueError('offset cannot be used without limit.')
            if not isinstance(offset, (int, np.integer)):
                raise TypeError('offset must be an integer.')
            comm += "OFFSET ? "
            args.append(int(offset))

        comm = comm + ';'

        if args == []:
            args = None
        res = self.execute(comm, args)
        return res

    def copy(self, indexes=None):
        """Get a copy of the database."""
        return self.__copy__(indexes=indexes)

    def column_names(self, table):
        """Get the column names of the table."""
        return self._column_cache[table].columns()

    def _query_colnames(self, table):
        """Query raw column names from the database."""
        self._check_table(table)
        comm = "SELECT * FROM "
        comm += f"{table} LIMIT 1;"
        self.execute(comm)
        return [i[0].lower() for i in self._cur.description
                if i[0].lower() != _ID_KEY.lower()]

    def _build_column_cache(self):
        """Build the column name cache."""
        for table in self.table_names:
            self._column_cache[table] = _SQLColumnCacher(self, table,
                                                         self._allow_encode)

    @property
    def db(self):
        """Get the database name."""
        return str(self._db)

    @property
    def table_names(self):
        """Get the table names in the database."""
        comm = "SELECT name FROM sqlite_master WHERE type='table';"
        return [i[0] for i in self.execute(comm) if i[0] != 'sqlite_sequence']

    def _check_table(self, table):
        """Check if the table exists in the database."""
        if table not in self.table_names:
            raise KeyError(f'Table "{table}" does not exist.')

    def _add_missing_columns(self, table, columns):
        """Add missing columns to the table."""
        existing = set(self.column_names(table))
        for col in [i for i in columns if i not in existing]:
            self.add_column(table, col)

    def _add_data_dict(self, table, data, add_columns=False,
                       skip_sanitize=False):
        """Add data sotred in a dict to the table."""
        data = _sanitize_colnames(data)
        if add_columns:
            self._add_missing_columns(table, data.keys())

        dict_row_list = _dict2row(cols=self.column_names(table), **data)
        try:
            rows = np.broadcast(*dict_row_list)
        except ValueError:
            rows = broadcast(*dict_row_list)
        rows = list(zip(*rows.iters))
        self._add_data_list(table, rows, skip_sanitize=skip_sanitize)

    def _add_data_list(self, table, data, skip_sanitize=False):
        """Add data stored in a list to the table."""
        if np.ndim(data) not in (1, 2):
            raise ValueError('data must be a 1D or 2D array.')

        if np.ndim(data) == 1:
            data = np.reshape(data, (1, len(data)))

        if np.shape(data)[1] != len(self.column_names(table)):
            raise ValueError('data must have the same number of columns as '
                             'the table.')

        if not skip_sanitize:
            data = [tuple(map(_sanitize_value, d)) for d in data]
        comm = f"INSERT INTO {table} VALUES "
        comm += f"(NULL, {', '.join(['?']*len(data[0]))})"
        comm += ';'
        self.executemany(comm, data)

        # Update the row indexes
        rl = self._row_indexes[table]
        rl.extend([_SQLRowIndexer(rl) for i in range(len(data))])

    def _get_indexes(self, table):
        """Get the indexes of the table."""
        comm = f"SELECT {_ID_KEY} FROM {table};"
        return [i[0] for i in self.execute(comm)]

    def _update_indexes(self, table):
        """Update the indexes of the table."""
        rows = list(range(1, self.count(table) + 1))
        origin = self._get_indexes(table)
        comm = f"UPDATE {table} SET {_ID_KEY} = ? WHERE {_ID_KEY} = ?;"
        self.executemany(comm, zip(rows, origin))

    def _build_row_indexes(self):
        """Build the row indexes."""
        for table in self.table_names:
            size = self.count(table)
            # Create the list that must be passed to _SQLRowIndexer
            rl = [None]*len(size)
            self._row_indexes[table] = rl
            for i in range(size):
                self._row_indexes[table][i] = _SQLRowIndexer(rl)

    def add_table(self, table, columns=None, data=None):
        """Create a table in database."""
        logger.debug('Initializing "%s" table.', table)
        if table in self.table_names:
            raise ValueError('table {table} already exists.')

        comm = f"CREATE TABLE '{table}'"
        comm += f" (\n{_ID_KEY} INTEGER PRIMARY KEY AUTOINCREMENT"

        if columns is not None and data is not None:
            raise ValueError('cannot specify both columns and data.')
        if columns is not None:
            comm += ",\n"
            for i, name in enumerate(columns):
                comm += f"\t'{name}'"
                if i != len(columns) - 1:
                    comm += ",\n"
        comm += "\n);"

        self.execute(comm)

        # Add the row indexer list
        self._row_indexes[table] = []

        # Add the column cache
        self._column_cache[table] = _SQLColumnCacher(self, table,
                                                     self._allow_encode)

        if data is not None:
            self.add_rows(table, data, add_columns=True)

    def add_column(self, table, column, data=None):
        """Add a column to a table."""
        self._check_table(table)

        column = column.lower()
        if data is not None and len(data) != len(self[table]) and \
           len(self[table]) != 0:
            raise ValueError("data must have the same length as the table.")

        if column in (_ID_KEY, 'table', 'default'):
            raise ValueError(f"{column} is a protected name.")

        col = _sanitize_colnames([column],
                                 allow_b32_encode=self._allow_encode)[0]
        comm = f"ALTER TABLE {table} ADD COLUMN '{col}' ;"
        logger.debug('adding column "%s" to table "%s"', col, table)
        self.execute(comm)

        # add the column to the cache
        self._column_cache[table].add_column(column)

        # adding the data to the table
        if data is not None:
            self.set_column(table, column, data)

    def delete_column(self, table, column):
        """Delete a column from a table."""
        self._check_table(table)
        column = self._column_cache[table].get_column(column)

        if column in (_ID_KEY, 'table', 'default'):
            raise ValueError(f"{column} is a protected name.")

        comm = f"ALTER TABLE {table} DROP COLUMN '{column}' ;"
        logger.debug('deleting column "%s" from table "%s"', column, table)
        self.execute(comm)
        self._column_cache[table].remove_column(column)

    def add_rows(self, table, data, add_columns=False, skip_sanitize=False):
        """Add a dict row to a table.

        Parameters
        ----------
        data : dict, list or `~numpy.ndarray`
            Data to add to the table. If dict, keys are column names,
            if list, the order of the values is the same as the order of
            the column names. If `~numpy.ndarray`, dtype names are interpreted
            as column names.
        add_columns : bool (optional)
            If True, add missing columns to the table.
        """
        self._check_table(table)
        if isinstance(data, (list, tuple)):
            return self._add_data_list(table, data,
                                       skip_sanitize=skip_sanitize)
        if isinstance(data, dict):
            return self._add_data_dict(table, data, add_columns=add_columns,
                                       skip_sanitize=skip_sanitize)
        if isinstance(data, np.ndarray):
            names = data.dtype.names
            if names is not None:
                data = {n: data[n] for n in names}
                return self._add_data_dict(table, data,
                                           add_columns=add_columns,
                                           skip_sanitize=skip_sanitize)
            return self._add_data_list(table, data,
                                       skip_sanitize=skip_sanitize)
        if isinstance(data, Table):
            data = {c: list(data[c]) for c in data.colnames}
            return self._add_data_dict(table, data, add_columns=add_columns,
                                       skip_sanitize=skip_sanitize)

        raise TypeError('data must be a dict, list, or numpy array. '
                        f'Not {type(data)}.')

    def delete_row(self, table, index):
        """Delete a row from the table."""
        self._check_table(table)
        row = _fix_row_index(index, len(self[table]))
        comm = f"DELETE FROM {table} WHERE {_ID_KEY}={row+1};"
        self.execute(comm)
        self._row_indexes[table].pop(row)
        self._update_indexes(table)

    def drop_table(self, table):
        """Drop a table from the database."""
        self._check_table(table)
        comm = f"DROP TABLE {table};"
        self.execute(comm)
        del self._row_indexes[table]

    def get_table(self, table):
        """Get a table from the database."""
        self._check_table(table)
        return SQLTable(self, table)

    def get_row(self, table, index):
        """Get a row from the table."""
        self._check_table(table)
        index = _fix_row_index(index, len(self[table]))
        row = self._row_indexes[table][index]
        return SQLRow(self, table, row)

    def get_column(self, table, column):
        """Get a column from the table."""
        column = self._column_cache[table].get_column(column)
        return SQLColumn(self, table, column)

    def get_item(self, table, column, row):
        """Get an item from the table."""
        self._check_table(table)
        row = _fix_row_index(row, len(self[table]))
        return self.get_column(table, column)[row]

    def set_item(self, table, column, row, value):
        """Set a value in a cell."""
        row = _fix_row_index(row, self.count(table))
        column = self._column_cache[table].get_column(column)
        value = _sanitize_value(value)
        self.execute(f"UPDATE {table} SET {column}=? "
                     f"WHERE {_ID_KEY}=?;", (value, row+1))

    def set_row(self, table, row, data):
        """Set a row in the table."""
        row = _fix_row_index(row, self.count(table))
        colnames = self.column_names(table)

        # TODO: use column caching
        if isinstance(data, dict):
            data = _dict2row(colnames, **data)
        elif isinstance(data, (list, tuple, np.ndarray)):
            if len(data) != len(colnames):
                raise ValueError('data must have the same length as the '
                                 'table.')
        else:
            raise TypeError('data must be a dict, list, or numpy array. '
                            f'Not {type(data)}.')

        comm = f"UPDATE {table} SET "
        comm += f"{', '.join(f'{i}=?' for i in colnames)} "
        comm += f" WHERE {_ID_KEY}=?;"

        self.execute(comm, tuple(list(map(_sanitize_value, data)) + [row+1]))

    def set_column(self, table, column, data):
        """Set a column in the table."""
        tablen = self.count(table)
        if column not in self.column_names(table):
            raise KeyError(f"column {column} does not exist.")
        if len(data) != tablen and tablen != 0:
            raise ValueError("data must have the same length as the table.")

        if tablen == 0:
            for i in range(len(data)):
                self.add_rows(table, {})

        col = self._column_cache[table].get_column(column)
        comm = f"UPDATE {table} SET "
        comm += f"{col}=? "
        comm += f" WHERE {_ID_KEY}=?;"
        args = list(zip([_sanitize_value(d) for d in data],
                        range(1, self.count(table)+1)))
        self.executemany(comm, args)

    def index_of(self, table, where):
        """Get the index(es) where a given condition is satisfied."""
        indx = self.select(table, _ID_KEY, where=where)
        if len(indx) == 1:
            return indx[0][0]-1
        return [i[0]-1 for i in indx]

    def __len__(self):
        """Get the number of rows in the current table."""
        return len(self.table_names)

    def __del__(self):
        """Delete the class, closing the db connection."""
        # ensure connection is closed.
        self._con.close()

    def __setitem__(self, item, value):
        """Set a row in the table."""
        if not isinstance(item, tuple):
            raise KeyError('item must be a in the formats '
                           'db[table, row], db[table, column] or '
                           'db[table, column, row].')
        if not isinstance(item[0], str):
            raise KeyError('first item must be the table name.')
        self.get_table(item[0])[item[1:]] = value

    def __getitem__(self, item):
        """Get a items from the table."""
        if isinstance(item, (str, np.str_)):
            return self.get_table(item)
        if isinstance(item, tuple):
            if not isinstance(item[0], str):
                raise ValueError('first item must be the table name.')
            return self.get_table(item[0])[item[1:]]
        raise ValueError('items must be a string for table names, '
                         'or a tuple in the formats. '
                         'db[table, row], db[table, column] or '
                         'db[table, column, row].')

    def __repr__(self):
        """Get a string representation of the table."""
        s = f"{self.__class__.__name__} '{self.db}' at {hex(id(self))}:"
        if len(self) == 0:
            s += '\n\tEmpty database.'
        for i in self.table_names:
            s += f"\n\t{i}: {len(self.column_names(i))} columns"
            s += f" {len(self[i])} rows"
        return s

    def __copy__(self, indexes=None):
        """Copy the database.

        Parameters
        ----------
        indexes : dict, optional
            A dictionary of table names and their indexes to copy.

        Returns
        -------
        db : SQLDatabase
            A copy of the database.
        """
        def _get_data(table, indx=None):
            if indx is None:
                return self.select(table)
            if len(indx) == 0:
                return None
            if len(indx) >= 10000:
                # too many rows must be divided
                indx = np.array_split(indx, np.ceil(len(indx)/10000))
                d = None
                for i in indx:
                    if d is None:
                        d = _get_data(table, i)
                    else:
                        np.vstack([d, _get_data(table, i)])
                return d
            where = " OR ".join(f"{_ID_KEY}={v+1}" for v in indx)
            return self.select(table, where=where)

        # when copying, always copy to memory
        db = SQLDatabase(':memory:')
        if indexes is None:
            indexes = {}
        for i in self.table_names:
            db.add_table(i, columns=self.column_names(i))
            rows = _get_data(i, indexes.get(i, None))
            if rows is not None:
                db.add_rows(i, rows, skip_sanitize=True)
        return db
