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
    for i, name in enumerate(cols):
        if name in data.keys():
            d = data[name]
            if isinstance(d, str):
                d = f"'{d}'"
            elif isinstance(d, bytes):
                d = f"'{d.decode()}'"
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


class Database:
    """Database creation and manipulation with SQL.

    Notes
    -----
    - __id__ is only for internal indexing. It is ignored on returns.
    """

    def __init__(self, db=':memory:', table='main', dtype=None, length=0,
                 data=None):
        self._db = db
        self._con = sql.connect(self._db)
        self._cur = self._con.cursor()
        self._table = table

        # add the main table
        self._add_table(self._table, dtype)

        if data is not None and length != 0:
            raise ValueError('data and length cannot be both set.')

        # initialize a length if needed
        for _ in range(length):
            self.add_row({})

        if data is not None:
            for row in _import_from_data(data):
                self.add_row(row, add_columns=True)

    def execute(self, command):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))
        self._cur.execute(command)
        res = self._cur.fetchall()
        self._con.commit()
        return res

    def _add_table(self, table, dtype):
        """Create a table in database."""
        logger.debug('Initializing "%s" table.', table)
        tables = [i[0] for i in self.execute("SELECT name FROM sqlite_master "
                                             "WHERE type='table';")]
        if table in tables:
            logger.debug('table "%s" already exists', table)
            return
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

    def colnames(self, table=None):
        """Get the column names of the current table."""
        comm = "SELECT * FROM "
        comm += f"{table or self._table} LIMIT 1;"
        self.execute(comm)
        return [i[0].lower() for i in self._cur.description
                if i[0].lower() != _ID_KEY.lower()]

    def values(self, table=None):
        """Get the values of the current table."""
        return self.select(table=table)

    def add_column(self, column, dtype=None, data=None, table=None):
        """Add a column to a table."""
        if data is not None and len(data) != self.__len__(table) and \
           self.__len__(table) != 0:
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
            self.set_column(column, data, table=table)

    def add_row(self, data, add_columns=False, table=None):
        """Add a dict row to a table.

        Parameters
        ----------
        data : dict
            Dictionary of data to add.
        add_columns : bool (optional)
            If True, add missing columns to the table.
        """
        data = _sanitize_colnames(data)

        if add_columns:
            # add missing columns
            cols = set(self.colnames(table=table))
            for k in data.keys():
                if k not in cols:
                    self.add_column(k, np.array([data[k]]).dtype,
                                    table=table)

        comm_dict = _row_dict(data, self.colnames(table=table))
        # create the sql command and add the row
        cols = [_ID_KEY] + self.colnames(table=table)
        comm = f"INSERT INTO {table or self._table} VALUES ("
        comm += f"{', '.join(comm_dict[i] for i in cols)});"
        self.execute(comm)

    def select(self, columns=None, where=None, table=None):
        """Select rows from a table.

        Parameters
        ----------
        columns : list (optional)
            List of columns to select. If None, select all columns.
        where : dict (optional)
            Dictionary of conditions to select rows. Keys are column names,
            values are values to compare. All rows equal to the values will
            be selected. If None, all rows are selected.
        """
        if columns is None:
            columns = self.colnames(table=table)
        # only use sanitized column names
        columns = ', '.join(_sanitize_colnames(columns))

        if where is None:
            _where = '1=1'
        else:
            for i, (k, v) in enumerate(where.items()):
                if isinstance(v, str):
                    # avoid sql errors
                    v = f"'{v}'"
                if i == 0:
                    _where = f"{k}={v}"
                else:
                    _where += f" AND {k}={v}"
        res = self.execute(f"SELECT {columns} FROM {table or self._table} "
                           f"WHERE {_where}")
        return res

    def __len__(self, table=None):
        """Get the number of rows in the current table."""
        comm = f"SELECT COUNT(*) FROM {table or self._table}"
        return self.execute(comm)[0][0]

    def __del__(self):
        """Delete the class, closing the db connection."""
        # ensure connection is closed.
        self._con.close()

    def as_table(self, table=None):
        """Return the current table as an `~astropy.table.Table` object."""
        if self.__len__(table=table) == 0:
            return Table(names=self.colnames(table=table))
        return Table(rows=self.select(table=table),
                     names=self.colnames(table=table))

    def get_row(self, index, table=None):
        """Get a row from the table."""
        index = _fix_row_index(index, self.__len__(table=table))
        return dict(zip(self.colnames(table=table),
                        self.select(where={_ID_KEY: index+1},
                                    table=table)[0]))

    def get_column(self, column, table=None):
        """Get a column from the table."""
        try:
            res = [i[0] for i in self.select(columns=[column], table=table)]
        except sql.OperationalError:
            raise KeyError(f"column '{column}' does not exist in table "
                           f"'{table or self._table}'")

        if len(res) == 1:
            return res[0]
        return res

    def set_item(self, column, row, value, table=None):
        """Set a value in a cell."""
        row = _fix_row_index(row, self.__len__(table=table))
        if isinstance(value, str):
            value = f"'{value}'"
        self.execute(f"UPDATE {table or self._table} SET {column}={value} "
                     f"WHERE {_ID_KEY}={row+1};")

    def set_row(self, row, data, table=None):
        """Set a row in the table."""
        row = _fix_row_index(row, self.__len__(table=table))
        comm_dict = _row_dict(data, self.colnames(table=table))
        comm = f"UPDATE {table or self._table} SET "
        for i, name in enumerate(self.colnames(table=table)):
            comm += f"{name}={comm_dict[name]}"
            if i != len(self.colnames(table=table)) - 1:
                comm += ", "
        comm += f" WHERE {_ID_KEY}={row+1};"

    def set_column(self, column, data, table=None):
        """Set a column in the table."""
        if len(data) != self.__len__(table=table) and \
           self.__len__(table=table) != 0:
            raise ValueError("data must have the same length as the table.")

        if self.__len__(table=table) == 0:
            for i in range(len(data)):
                self.add_row({}, table=table)

        col = _sanitize_colnames([column])[0]
        for i, d in enumerate(data):
            if isinstance(d, str):
                d = f"'{d}'"
            comm = f"UPDATE {table or self._table} SET {col} = {d} "
            comm += f"WHERE {_ID_KEY} = {i+1};"
            self.execute(comm)

    def __setitem__(self, item, value):
        """Set a row in the table."""
        raise NotImplementedError('use set_item, set_col or set_row instead.')

    def __getitem__(self, item, table=None):
        """Get a items from the table."""
        table = table or self._table
        if isinstance(item, int):
            db = Database(':memory:', table=table)
            db.add_row(self.get_row(item, table=table),
                       add_columns=True)
            return db

        elif isinstance(item, str):
            return self.get_column(item, table=table)

        elif isinstance(item, slice):
            db = Database(':memory:', table=table)
            # Python allow slicing outside the range of the table
            rows = self.values(table=table)[item]
            if len(rows) == 0:
                for c in self.colnames(table=table):
                    dt = np.array(self.get_column(c, table=table)).dtype
                    db.add_column(c, dtype=dt)
            else:
                for row in rows:
                    db.add_row(dict(zip(self.colnames(table=table), row)),
                               add_columns=True)
            return db

        elif isinstance(item, (list, np.ndarray)):
            if isinstance(item[0], str):
                db = Database(':memory:', table=table, length=len(self))
                for i in item:
                    db.add_column(i, data=self.__getitem__(i, table=table))
            else:
                db = Database(':memory:', table=table)
                for i in item:
                    db.add_row(self.get_row(i, table=table), add_columns=True)
            return db

    def __repr__(self):
        """Get a string representation of the table."""
        s = f"{self.__class__.__name__} at {hex(id(self))}:"
        s += f" table '{self._table}' "
        s += f"({len(self.colnames())} columns x {len(self)} rows)\n"
        s += '\n'.join(self.as_table().__repr__().split('\n')[1:])
        return s
