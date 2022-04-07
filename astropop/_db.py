# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage SQL databases in a simplier way."""

import sqlite3 as sql


np_to_sql = {
    'i': 'INTEGER',
    'f': 'REAL',
    'S': 'TEXT',
    'U': 'TEXT',
    'b': 'BOOLEAN',
}


class DataBase:
    """Database creation and manipulation with SQL."""

    def __init__(self, db):
        self._db = db
        self._con = sql.connect(self._db)
        self._cur = self._con.cursor()

    def execute(self, command):
        """Execute a SQL command in the database."""
        print('executing:\n', command)
        self._cur.execute(command)
        res = self._cur.fetchall()
        self._con.commit()
        return res

    def add_table(self, table, dtype=None):
        """Create a table in database."""
        comm = f"CREATE TABLE {table}"
        if dtype is not None:
            comm += " (\n"
            for i, name in enumerate(dtype.names):
                kind = dtype[i].kind
                comm += f"\t{name} {np_to_sql[kind]}"
                if i != len(dtype) - 1:
                    comm += ",\n"
            comm += "\n);"
        return self.execute(comm)

    def colnames(self, table=None):
        """Get the column names of the current table."""
        if table is not None:
            self.execute(f"SELECT * FROM {table} WHERE 1=0")
        return [i[0] for i in self._cur.description]

    def add_column(self, table, column, dtype):
        """Add a column to a table."""

    def add_row(self, table, data):
        """Add a row to a table."""

    def select(self, table, columns=None, where=None):
        """Select rows from a table."""

    def __del__(self):
        """Delete the class, closing the db connection."""
        # ensure connection is closed.
        self._con.close()
