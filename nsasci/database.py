import pandas as pd
import numpy as np
import sqlite3

class NsaSciDatabase(object):
    def __init__(self, path2db):
        self.db = sqlite3.connect(path2db)


    def add_line2db(self, line, table_name, if_exists = 'fail'):
        """ line is dataframe with same colum names
           """
        idx = line.index[0]
        idx_label = line.index.name
    #     print(idx)
    #     print(idx_label)
        qu = 'SELECT EXISTS(SELECT 1 FROM {} WHERE {}="{}");'.format(table_name, idx_label, idx)
        exists = self.db.execute(qu).fetchall()

        if np.any(np.array(exists) == 1):
            if if_exists == 'fail':
                raise ValueError('Index {} exists'.format(idx))
            elif if_exists == 'skip':
                print('Idx {} exists ... skipping'.format(idx))
                return False
            elif if_exists == 'overwrite':
                raise ValueError('not implemented yet ... maybe its time')

        # create/append table

        line.to_sql(table_name, self.db,
        #                  if_exists='replace'
                         if_exists='append'
                        )

    def add_table2db(self, columns=['fname','date','start', 'end'], table_name = 'imet', index_name = 'idx', if_exists='fail'):
        """This function will add a table to the database
        if_exists: str ['fail', 'replace']"""
        flight_df = pd.DataFrame(columns=columns)
        if index_name:
            flight_df.index.name = index_name
        else:
            flight_df.index.name = 'idx'
        tbl_name = table_name
        flight_df.to_sql(tbl_name, self.db,
                             if_exists=if_exists
        #                      if_exists='append'
                            )

    def dump_table(self, tbl_name, index_col = None):
        qu = 'select * from {}'.format(tbl_name)
        df_db = pd.read_sql(qu, self.db, index_col= index_col)
        return df_db