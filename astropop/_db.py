# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage SQL databases in a simplier way."""

import sqlite3 as sql
import numpy as np
from astropy.table import Table

from .logger import logger
from .py_utils import check_iterable


__all__ = ['SQLDatabase', 'SQLTable', 'SQLRow', 'SQLColumn']


np_to_sql = {
    'i': 'INTEGER',
    'f': 'REAL',
    'S': 'TEXT',
    'U': 'TEXT',
    'b': 'BOOLEAN',
}


_ID_KEY = '__id__'


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
    if isinstance(data, str):
        return _sanitize(data)
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f'{type(data)} is not supported.')

    return [_sanitize(i).lower() for i in data]


def _fix_row_index(row, length):
    """Fix the row number to be a valid index."""
    if row < 0:
        row += length
    if row >= length or row < 0:
        raise IndexError('Row index out of range.')
    return row


def _row_dict(data, cols):
    """Convert a dict to match the colnames fo a database."""
    if not isinstance(data, dict):
        raise TypeError(f'{type(data)} is not supported.')
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
            for row in [dict(zip(data.keys(), i))
                        for i in zip(*data.values())]:
                yield row
        else:
            yield data
    else:
        raise TypeError(f'{type(data)} is not supported.')


def _parse_where(where):
    if where is None:
        _where = None
    elif isinstance(where, dict):
        where = _sanitize_colnames(where)
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

    def select(self, *args, **kwargs):
        """Select rows from the table."""
        return self._db.select(self._name, *args, **kwargs)

    def as_table(self):
        """Return the current table as an `~astropy.table.Table` object."""
        if len(self) == 0:
            return Table(names=self.column_names)
        return Table(rows=self.values,
                     names=self.column_names)

    def add_column(self, name, data=None):
        """Add a column to the table."""
        self._db.add_column(self._name, name, data=data)

    def add_row(self, data, add_columns=False):
        """Add a row to the table."""
        self._db.add_row(self._name, data, add_columns=add_columns)

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
        return self._db.select(self._table)[self.index]

    @property
    def index(self):
        """Get the index of the current row."""
        return self._row

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
            try:
                return self._db.get_item(self._table, key, self._row)
            except ValueError:
                raise KeyError(f'{key}')
        if isinstance(key, (int, np.int_)):
            return self.values[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key, value):
        """Set a column in the row."""
        if key not in self.column_names:
            raise KeyError(f'{key}')
        self._db.set_item(self._table, key, self.index, value)

    def __iter__(self):
        """Iterate over the row."""
        for i in self.values:
            yield i

    def __contains__(self, item):
        """Check if the row contains a given value."""
        return item in self.values

    def __repr__(self):
        """Get a string representation of the row."""
        s = f"{self.__class__.__name__} {self._row} in table '{self._table}' "
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
        self.autocommit = autocommit

    def execute(self, command):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))
        self._cur.execute(command)
        res = self._cur.fetchall()
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

    def copy(self):
        """Get a copy of the database."""
        return self.__copy__()

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
    def table_names(self):
        """Get the table names in the database."""
        comm = "SELECT name FROM sqlite_master WHERE type='table';"
        return [i[0] for i in self.execute(comm) if i[0] != 'sqlite_sequence']

    def _check_table(self, table):
        """Check if the table exists in the database."""
        if table not in self.table_names:
            raise KeyError(f'Table "{table}" does not exist.')

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
        if data is not None:
            rows = list(_import_from_data(data))

        self.execute(comm)

        if data is not None:
            for r in rows:
                self.add_row(table, r, add_columns=True)

    def add_column(self, table, column, data=None):
        """Add a column to a table."""
        self._check_table(table)

        column = column.lower()
        if data is not None and len(data) != len(self[table]) and \
           len(self[table]) != 0:
            raise ValueError("data must have the same length as the table.")

        if column in (_ID_KEY, 'table', 'default'):
            raise ValueError(f"{column} is a protected name.")

        col = _sanitize_colnames([column])[0]
        comm = f"ALTER TABLE {table or self._table} ADD COLUMN '{col}' ;"
        logger.debug('adding column "%s" "%s" "%s" to table "%s"',
                     col, table or self._table)
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
        if not isinstance(data, dict):
            raise TypeError('data must be a dict.')
        self._check_table(table)
        data = _sanitize_colnames(data)

        if add_columns:
            # add missing columns
            cols = set(self.column_names(table))
            for k in data.keys():
                if k not in cols:
                    self.add_column(table, k)

        comm_dict = _row_dict(data, self.column_names(table))
        # create the sql command and add the row
        cols = [_ID_KEY] + self.column_names(table)
        comm = f"INSERT INTO {table} VALUES ("
        comm += f"{', '.join(comm_dict[i] for i in cols)});"
        self.execute(comm)

    def drop_table(self, table):
        """Drop a table from the database."""
        self._check_table(table)
        comm = f"DROP TABLE {table};"
        self.execute(comm)

    def get_table(self, table):
        """Get a table from the database."""
        self._check_table(table)
        return SQLTable(self, table)

    def get_row(self, table, index):
        """Get a row from the table."""
        self._check_table(table)
        index = _fix_row_index(index, len(self[table]))
        return SQLRow(self, table, index)

    def get_column(self, table, column):
        """Get a column from the table."""
        column = column.lower()
        if column not in self.column_names(table):
            raise KeyError(f"column {column} does not exist.")
        return SQLColumn(self, table, column)

    def get_item(self, table, column, row):
        """Get an item from the table."""
        self._check_table(table)
        row = _fix_row_index(row, len(self[table]))
        return self.get_column(table, column)[row]

    def set_item(self, table, column, row, value):
        """Set a value in a cell."""
        row = _fix_row_index(row, self.count(table))
        column = column.lower()
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
        comm += f"{', '.join(f'{i}={comm_dict[i]}' for i in colnames)} "
        comm += f" WHERE {_ID_KEY}={row+1};"
        self.execute(comm)

    def set_column(self, table, column, data):
        """Set a column in the table."""
        tablen = self.count(table)
        if column not in self.column_names(table):
            raise KeyError(f"column {column} does not exist.")
        if len(data) != tablen and tablen != 0:
            raise ValueError("data must have the same length as the table.")

        if tablen == 0:
            for i in range(len(data)):
                self.add_row(table, {})

        col = _sanitize_colnames([column])[0]
        for i, d in enumerate(data):
            self.set_item(table, col, i, d)

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

    def __copy__(self):
        """Copy the database."""
        # when copying, always copy to memory
        db = SQLDatabase(':memory:')
        for i in self.table_names:
            db.add_table(i, data=self[i].as_table())
        return db
