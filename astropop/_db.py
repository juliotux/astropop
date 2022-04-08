# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage SQL databases in a simplier way."""

import sqlite3 as sql
import numpy as np

from .logger import logger


np_to_sql = {
    'i': 'INTEGER',
    'f': 'REAL',
    'S': 'TEXT',
    'U': 'TEXT',
    'b': 'BOOLEAN',
}


def _sanitize_colnames(colnames):
    """Sanitize the colnames to avoid invalid characteres like '-'."""
    return [c.replace('-', '_') for c in colnames]


class Database:
    """Database creation and manipulation with SQL."""

    def __init__(self, db, table='table', dtype=None):
        self._db = db
        self._con = sql.connect(self._db)
        self._cur = self._con.cursor()
        self._table = table
        self._add_table(dtype)

    def execute(self, command):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))
        self._cur.execute(command)
        res = self._cur.fetchall()
        self._con.commit()
        return res

    def _add_table(self, dtype):
        """Create a table in database."""
        logger.debug('Initializing "%s" table.', self._table)
        tables = [i[0] for i in self.execute("SELECT name FROM sqlite_master "
                                             "WHERE type='table';")]
        if self._table in tables:
            logger.debug('table "%s" already exists', self._table)
            return
        comm = f"CREATE TABLE {self._table}"
        comm += " (\n_id INTEGER PRIMARY KEY AUTOINCREMENT"
        if dtype is not None:
            comm += ",\n"
            for i, name in enumerate(dtype.names):
                kind = dtype[i].kind
                comm += f"\t'{name}' {np_to_sql[kind]}"
                if i != len(dtype) - 1:
                    comm += ",\n"
        comm += "\n);"
        self.execute(comm)

    def colnames(self):
        """Get the column names of the current table."""
        self.execute(f"SELECT * FROM {self._table} WHERE 1=0")
        return [i[0] for i in self._cur.description]

    def add_column(self, column, dtype):
        """Add a column to a table."""
        col = _sanitize_colnames([column])[0]
        comm = f"ALTER TABLE {self._table} ADD COLUMN '{col}' "
        logger.debug('adding column "%s" "%s" "%s"', col, dtype, dtype.kind)
        comm += f"{np_to_sql[dtype.kind]};"
        self.execute(comm)

    def add_row(self, data, add_columns=False):
        """Add a dict row to a table.

        Parameters
        ----------
        data : dict
            Dictionary of data to add.
        add_columns : bool (optional)
            If True, add missing columns to the table.
        """
        data_c = {}
        sanitized = _sanitize_colnames(data.keys())
        # create a dict copy with sanitized keys
        for k, ks in zip(data.keys(), sanitized):
            data_c[ks] = data[k]

        if add_columns:
            # add missing columns
            cols = set(self.colnames())
            for k in data_c.keys():
                if k not in cols:
                    self.add_column(k, np.array([data_c[k]]).dtype)

        # create the sql command and add the row
        cols = self.colnames()
        comm = f"INSERT INTO {self._table} VALUES ("
        for i, name in enumerate(cols):
            if name in data_c.keys():
                d = data_c[name]
                if isinstance(d, str):
                    d = f"'{d}'"
                comm += f"{d}"
            elif name == '_id':
                comm += "NULL"
            else:
                comm += "NULL"
            if i != len(cols) - 1:
                comm += ", "
        comm += ");"
        self.execute(comm)

    def select(self, columns=None, where=None):
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
            columns = '*'
        else:
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
        return self.execute(f"SELECT {columns} FROM {self._table} "
                            f"WHERE {_where}")

    def __len__(self):
        """Get the number of rows in the current table."""
        return self.execute(f"SELECT COUNT(*) FROM {self._table}")[0][0]

    def __del__(self):
        """Delete the class, closing the db connection."""
        # ensure connection is closed.
        self._con.close()
