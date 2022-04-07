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


class DataBase:
    """Database creation and manipulation with SQL."""

    def __init__(self, db, table='table', dtype=None):
        self._db = db
        self._con = sql.connect(self._db)
        self._cur = self._con.cursor()
        self._table = table
        self._dtype = dtype
        self._add_table()

    def execute(self, command):
        """Execute a SQL command in the database."""
        logger.debug('executing sql command: "%s"',
                     str.replace(command, '\n', ' '))
        self._cur.execute(command)
        res = self._cur.fetchall()
        self._con.commit()
        return res

    def _add_table(self):
        """Create a table in database."""
        logger.debug('Initializing "%s" table.', self._table)
        tables = [i[0] for i in self.execute("SELECT name FROM sqlite_master "
                                             "WHERE type='table';")]
        if self._table in tables:
            logger.debug('table "%s" already exists', self._table)
            return
        comm = f"CREATE TABLE {self._table}"
        comm += " (\n_id INTEGER PRIMARY KEY AUTOINCREMENT"
        if self._dtype is not None:
            comm += ",\n"
            for i, name in enumerate(self._dtype.names):
                kind = self._dtype[i].kind
                comm += f"\t{name} {np_to_sql[kind]}"
                if i != len(self._dtype) - 1:
                    comm += ",\n"
        comm += "\n);"
        self.execute(comm)

    def colnames(self):
        """Get the column names of the current table."""
        self.execute(f"SELECT * FROM {self._table} WHERE 1=0")
        return [i[0] for i in self._cur.description]

    def add_column(self, column, dtype):
        """Add a column to a table."""
        comm = f"ALTER TABLE {self._table} ADD COLUMN {column} {np_to_sql[dtype.kind]}"
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
        if add_columns:
            for i in data.keys():
                if i not in self.colnames():
                    self.add_column(i, np.array([data[i]]).dtype)

        comm = f"INSERT INTO {self._table} VALUES ("
        for i, name in enumerate(self._dtype.names):
            kind = self._dtype[i].kind
            comm += f"{data[name]}"
            if i != len(self._dtype) - 1:
                comm += ", "
        comm += ");"
        self.execute(comm)

    def select(self, columns=None, where=None):
        """Select rows from a table.

        Parameters
        ----------
        columns : list (optional)
            List of columns to select.
        where : dict (optional)
            Dictionary of conditions to select rows. Keys are column names,
            values are values to compare. All rows equal to the values will
            be selected. If None, all rows are selected.
        """
        if columns is None:
            columns = '*'
        else:
            columns = ', '.join(columns)
        if where is None:
            _where = '1=1'
        else:
            for i, (k, v) in enumerate(where.items()):
                if isinstance(v, str):
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
