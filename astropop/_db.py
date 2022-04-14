# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage SQL databases in a simplier way."""

import sqlite3 as sql
import numpy as np
from astropy.table import Table

from .logger import logger
from .py_utils import check_iterable


np_to_sql = {
    'i': 'INTEGER',
    'f': 'REAL',
    'S': 'TEXT',
    'U': 'TEXT',
    'b': 'BOOLEAN',
}


_ID_KEY = '__id__'


# TODO: redesign:
# - db[table, column, row], db[table, column], db[table, row] only
#   - works for getting and setting.
# - table as the first argument of all functions.
# - decorator to check if table exists.
# - only unitialized db, without tables.
# - add_table as a public function.


def _sanitize_colnames(data):
    """Sanitize the colnames to avoid invalid characteres like '-'."""
    def _sanitize(key):
        non_alpha = [ch for ch in key if not ch.isalnum()]
        for i in non_alpha:
            key = key.replace(i, '_')
        return key

    if isinstance(data, dict):
        d = data
        colnames = _sanitize_colnames(list(data.keys()))
        return dict(zip(colnames, d.values()))
    elif isinstance(data, str):
        return _sanitize(data)
    elif not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f'{type(data)} is not supported.')

    return [_sanitize(i) for i in data]


def _fix_row_index(row, length):
    """Fix the row number to be a valid index."""
    if row < 0:
        row += length
    if row > length or row < 0:
        raise IndexError('Row index out of range.')
    return row


def _row_dict(data, cols):
    """Convert a dict to match the colnames fo a database."""
    data = _sanitize_colnames(data)
    comm_dict = {_ID_KEY: "NULL"}
    for name in cols:
        if name in data.keys():
            d = data[name]
            if isinstance(d, str):
                d = f"'{d}'"
            elif isinstance(d, bytes):
                d = f"'{d.decode()}'"
            elif d is None:
                d = 'NULL'
            comm_dict[name] = f"{d}"
        else:
            comm_dict[name] = "NULL"
    return comm_dict


def _import_from_data(data):
    """Import data from a dict or a list of dicts."""
    if isinstance(data, Table):
        data = data.as_array()

    if isinstance(data, np.ndarray):
        for i in data:
            yield dict(zip(data.dtype.names, i))
    elif isinstance(data, dict):
        if np.any([check_iterable(i) for i in data.values()]):
            for row in [dict(zip(data.keys(), i)) for i in zip(*data.values())]:
                yield row
        else:
            yield data
    else:
        raise TypeError(f'{type(data)} is not supported.')


def _parse_where(where):
    if where is None:
        _where = None
    elif isinstance(where, dict):
        for i, (k, v) in enumerate(where.items()):
            if isinstance(v, str):
                # avoid sql errors
                v = f"'{v}'"
            if i == 0:
                _where = f"{k}={v}"
            else:
                _where += f" AND {k}={v}"
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
    return _where


class SQLTable:
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
        return self._db.column_names(self._name)

    @property
    def values(self):
        """Get the values of the current table."""
        return self.select()

    def select(self, columns=None, where=None, order=None, limit=None):
        """Select rows from the table."""
        return self._db.select(self._name, columns, where, order, limit)

    def as_table(self):
        """Return the current table as an `~astropy.table.Table` object."""
        if len(self) == 0:
            return Table(names=self.column_names)
        return Table(rows=self.values,
                     names=self.column_names)

    def add_column(self, name, dtype=None, data=None):
        """Add a column to the table."""
        self._db.add_column(self._name, name, dtype=dtype, data=data)

    def add_row(self, data):
        """Add a row to the table."""
        self._db.add_row(self._name, data)

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

    def __getitem__(self, key):
        """Get a row or a column from the table."""
        if isinstance(key, int):
            return self.get_row(key)
        elif isinstance(key, str):
            return self.get_column(key)
        elif isinstance(key, (tuple, slice, list, np.ndarray)):
            raise NotImplementedError('TODO')
        else:
            raise KeyError(f'{key}')

    def __setitem__(self, key, value):
        """Set a row or a column in the table."""
        if isinstance(key, int):
            self.set_row(key, value)
        elif isinstance(key, str):
            self.set_column(key, value)
        elif isinstance(key, (tuple, slice, list, np.ndarray)):
            raise NotImplementedError('TODO')
        else:
            raise KeyError(f'{key}')

    def __len__(self):
        """Get the number of rows in the table."""
        return self._db.count(self._name)

    def __contains__(self, item):
        """Check if a given column is in the table."""
        if isinstance(item, str):
            return item in self.column_names
        else:
            raise TypeError(f'{item} is not supported.')

    def __repr__(self):
        """Get a string representation of the table."""
        s = f"{self.__class__.__name__} '{self.name}' at {hex(id(self))}:"
        s += f"({len(self.column_names)} columns x {len(self)} rows)\n"
        s += '\n'.join(self.as_table().__repr__().split('\n')[1:])
        return s


class SQLColumn:
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
        if isinstance(key, (int, tuple, slice, list, np.ndarray)):
            return self.values[key]
        else:
            raise KeyError(f'{key}')

    def __setitem__(self, key, value):
        """Set a row in the column."""
        if isinstance(key, int):
            self._db.set_item(self._table, self._name, key, value)
        else:
            raise KeyError(f'{key}')

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
        s += f" ({len(self)} rows):"
        return s


class SQLRow:
    """Handle and SQL table row interfacing with the DB."""

    def __init__(self, db, table, row):
        """Initialize the row.

        Parameters
        ----------
        db : SQLDatabase
            The parent database object.
        table : str
            The name of the table in the database.
        row : int
            The row index in the table.
        """
        self._db = db
        self._table = table
        self._row = row

    @property
    def column_names(self):
        """Get the column names of the current table."""
        return self._db.column_names(self._table)

    @property
    def table(self):
        """Get the table name."""
        return self._table

    @property
    def values(self):
        """Get the values of the current row."""
        return self._db.select(self._table)[self._row]

    @property
    def index(self):
        """Get the index of the current row."""
        return self._row

    @property
    def keys(self):
        """Get the keys of the current row."""
        return self.column_names

    def as_dict(self):
        """Get the row as a dict."""
        return dict(zip(self.column_names, self.values))

    def __getitem__(self, key):
        """Get a column from the row."""
        if isinstance(key, str):
            return self.values[self.column_names.index(key)]
        elif isinstance(key, int):
            return self.values[key]
        else:
            raise KeyError(f'{key}')

    def __setitem__(self, key, value):
        """Set a column in the row."""
        if key not in self.column_names:
            raise KeyError(f'{key}')
        self._db.set_item(self._table, key, self._row, value)

    def __iter__(self):
        """Iterate over the row."""
        for i in self.values:
            yield i

    def __contains__(self, item):
        """Check if the row contains a given value."""
        return item in self.values

    def __repr__(self):
        """Get a string representation of the row."""
        s = f"{self.__class__.__name__} {self._row} in table '{self._table}'"
        s += self.as_dict().__repr__()
        return s


class SQLDatabase:
    """Database creation and manipulation with SQL.

    Notes
    -----
    - '__id__' is only for internal indexing. It is ignored on returns.
    """

    def __init__(self, db=':memory:', autocommit=True):
        """Initialize the database.

        Parameters
        ----------
        db : str
            The name of the database file. If ':memory:' is given, the
            database will be created in memory.
        autocommit : bool (optional)
            Whether to commit changes to the database after each operation.
            Defaults to True.
        """
        self._db = db
        self._con = sql.connect(self._db)
        self._cur = self._con.cursor()
        self._autocommit = autocommit

    def execute(self, command):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))
        self._cur.execute(command)
        res = self._cur.fetchall()
        if self._autocommit:
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
        where = _parse_where(where)
        if where is not None:
            comm += f"WHERE {where}"
        comm += ";"
        return self.execute(comm)[0][0]

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
        # only use sanitized column names
        columns = ', '.join(_sanitize_colnames(columns))

        comm = f"SELECT {columns} "
        comm += f"FROM {table} "


        where = _parse_where(where)
        if where is not None:
            comm += f"WHERE {where} "

        if order is not None:
            comm += f"ORDER BY {order} ASC "

        if limit is not None:
            comm += f"LIMIT {limit} "
        if offset is not None:
            if limit is None:
                raise ValueError('offset cannot be used without limit.')
            comm += f"OFFSET {offset} "

        comm = comm + ';'

        res = self.execute(comm)
        return res

    def column_names(self, table):
        """Get the column names of the table."""
        self._check_table(table)
        comm = "SELECT * FROM "
        comm += f"{table} LIMIT 1;"
        self.execute(comm)
        return [i[0].lower() for i in self._cur.description
                if i[0].lower() != _ID_KEY.lower()]

    @property
    def db(self):
        """Get the database name."""
        return str(self._db)

    @property
    def autocommit(self):
        """Get whether the database commits after each operation."""
        return bool(self._autocommit)

    @property
    def table_names(self):
        """Get the table names in the database."""
        comm = "SELECT name FROM sqlite_master WHERE type='table';"
        return [i[0] for i in self.execute(comm) if i[0] != 'sqlite_sequence']

    def _check_table(self, table):
        """Check if the table exists in the database."""
        if table not in self.table_names:
            raise ValueError(f'Table "{table}" does not exist.')

    def add_table(self, table, dtype=None, data=None):
        """Create a table in database."""
        # TODO: handle data and dtype
        logger.debug('Initializing "%s" table.', table)
        if table in self.table_names:
            raise ValueError('table {table} already exists.')

        comm = f"CREATE TABLE '{table}'"
        comm += f" (\n{_ID_KEY} INTEGER PRIMARY KEY AUTOINCREMENT"
        if dtype is not None:
            comm += ",\n"
            for i, name in enumerate(dtype.names):
                kind = dtype[i].kind
                comm += f"\t'{name}' {np_to_sql[kind]}"
                if i != len(dtype) - 1:
                    comm += ",\n"
        comm += "\n);"

        self.execute(comm)

    def add_column(self, table, column, dtype=None, data=None):
        """Add a column to a table."""
        self._check_table(table)
        if data is not None and len(data) != len(self[table]) and \
           len(self[table]) != 0:
            raise ValueError("data must have the same length as the table.")

        if column in (_ID_KEY, 'table', 'default'):
            raise ValueError(f"{column} is a protected name.")

        # adding the column to the table
        if dtype is None and data is None:
            kind = ''
        elif dtype is not None:
            kind = np_to_sql[np.dtype(dtype).kind]
        else:
            kind = np_to_sql[np.array(data).dtype.kind]
        col = _sanitize_colnames([column])[0]
        comm = f"ALTER TABLE {table or self._table} ADD COLUMN '{col}' "
        logger.debug('adding column "%s" "%s" "%s" to table "%s"',
                     col, dtype, kind, table or self._table)
        comm += f"{kind};"
        self.execute(comm)

        # adding the data to the table
        if data is not None:
            self.set_column(table, column, data)

    def add_row(self, table, data, add_columns=False):
        """Add a dict row to a table.

        Parameters
        ----------
        data : dict
            Dictionary of data to add.
        add_columns : bool (optional)
            If True, add missing columns to the table.
        """
        self._check_table(table)
        data = _sanitize_colnames(data)

        if add_columns:
            # add missing columns
            cols = set(self.column_names(table))
            for k in data.keys():
                if k not in cols:
                    self.add_column(k, np.array([data[k]]).dtype,
                                    table=table)

        comm_dict = _row_dict(data, self.column_names(table))
        # create the sql command and add the row
        cols = [_ID_KEY] + self.column_names(table)
        comm = f"INSERT INTO {table} VALUES ("
        comm += f"{', '.join(comm_dict[i] for i in cols)});"
        self.execute(comm)

    def __len__(self):
        """Get the number of rows in the current table."""
        return len(self.table_names)

    def __del__(self):
        """Delete the class, closing the db connection."""
        # ensure connection is closed.
        self._con.close()

    def get_table(self, table):
        """Get a table from the database."""
        self._check_table(table)
        return SQLTable(self, table)

    def get_row(self, table, index):
        """Get a row from the table."""
        self._check_table(table)
        if index >= self.count(table):
            raise IndexError('index out of range.')
        return SQLRow(self, table, index)

    def get_column(self, table, column):
        """Get a column from the table."""
        if column not in self.column_names(table):
            raise KeyError(f"column {column} does not exist.")
        return SQLColumn(self, table, column)

    def set_item(self, table, column, row, value):
        """Set a value in a cell."""
        row = _fix_row_index(row, self.count(table))
        if isinstance(value, str):
            value = f"'{value}'"
        self.execute(f"UPDATE {table} SET {column}={value} "
                     f"WHERE {_ID_KEY}={row+1};")

    def set_row(self, table, row, data):
        """Set a row in the table."""
        row = _fix_row_index(row, self.count(table))
        colnames = self.column_names(table)

        comm_dict = _row_dict(data, colnames)

        comm = f"UPDATE {table} SET "
        for i, name in enumerate(colnames):
            comm += f"{name}={comm_dict[name]}"
            if i != len(colnames) - 1:
                comm += ", "
        comm += f" WHERE {_ID_KEY}={row+1};"
        self.execute(comm)

    def set_column(self, table, column, data):
        """Set a column in the table."""
        tablen = self.count(table)
        if len(data) != tablen and tablen != 0:
            raise ValueError("data must have the same length as the table.")

        if tablen == 0:
            for i in range(len(data)):
                self.add_row(table, {})

        col = _sanitize_colnames([column])[0]
        for i, d in enumerate(data):
            if isinstance(d, str):
                d = f"'{d}'"
            comm = f"UPDATE {table} SET {col} = {d} "
            comm += f"WHERE {_ID_KEY} = {i+1};"
            self.execute(comm)

    def __setitem__(self, item, value):
        """Set a row in the table."""
        if not isinstance(item, tuple):
            raise ValueError('item must be a in the formats '
                             'db[table, row], db[table, column] or '
                             'db[table, column, row].')
        raise NotImplementedError('TODO')

    def __getitem__(self, item):
        """Get a items from the table."""
        if isinstance(item, str):
            return self.get_table(item)
        elif isinstance(item, tuple):
            if not isinstance(item[0], str):
                raise ValueError('first item must be the table name.')
            return self.get_table(item[0])[item[1:]]
        else:
            raise ValueError('items must be a string for table names, '
                             'or a tuple in the formats. '
                             'db[table, row], db[table, column] or '
                             'db[table, column, row].')

    def __repr__(self):
        """Get a string representation of the table."""
        s = f"{self.__class__.__name__} at {hex(id(self))}:"
        if len(self) == 0:
            s += '\tEmpty table.'
        for i in self.table_names:
            s += f"\n\t{i}: {len(self.column_names(i))} columns"
            s += f" {len(self[i])} rows"
        return s
