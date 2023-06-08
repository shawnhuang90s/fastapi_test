# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:04
import asyncio
from aiomysql import Pool
from aiomysql import DictCursor
from aiomysql.connection import connect


class MySQLConfig:
    MIN_CACHED = 1
    MAX_CACHED = 5
    MINSIZE = 1
    MAXSIZE = 10
    POOL_RECYCLE = 3600
    HOST = "localhost"
    USER = "test_user"
    PASSWORD = "test_pwd"
    DB = "fastapi_test"
    PORT = 3306


class MySQLConn(object):
    pool = None

    def __init__(self, loop=asyncio.get_event_loop()):
        self._conn = None
        self._cur = None

    async def select_one(self, sql, params=None):
        try:
            count = await self._cur.execute(sql, params)
            if count > 0:
                return await self._cur.fetchone()
            else:
                return None
        except Exception as e:
            await self.rollback()
            raise e

    async def select_one_value(self, sql, params=None):
        try:
            count = await self._cur.execute(sql, params)
            if count > 0:
                result = await self._cur.fetchone()
                return list(result.values())[0]
            else:
                return None
        except Exception as e:
            await self.rollback()
            raise e

    async def select_many(self, sql, params=None):
        try:
            count = await self._cur.execute(sql, params)
            if count > 0:
                return await self._cur.fetchall()
            else:
                return []
        except Exception as e:
            await self.rollback()
            raise e

    async def select_many_one_value(self, sql, params=None):
        try:
            count = await self._cur.execute(sql, params)
            if count > 0:
                result = await self._cur.fetchall()
                return list(map(lambda one: list(one.values())[0], result))
            else:
                return []
        except Exception as e:
            await self.rollback()
            raise e

    async def insert_one(self, sql, params=None, return_auto_increament_id=False):
        try:
            result = await self._cur.execute(sql, params)
            if return_auto_increament_id:
                result = self._cur.lastrowid
            return result
        except Exception as e:
            await self.rollback()
            raise e

    async def insert_many(self, sql, params):
        try:
            count = await self._cur.executemany(sql, params)
            return count
        except Exception as e:
            await self.rollback()
            raise e

    async def update(self, sql, params=None):
        try:
            result = await self._cur.execute(sql, params)
            return result
        except Exception as e:
            await self.rollback()
            raise e

    async def delete(self, sql, params=None):
        try:
            result = await self._cur.execute(sql, params)
            return result
        except Exception as e:
            await self.rollback()
            raise e

    async def begin(self):
        await self._conn.begin()

    async def commit(self):
        try:
            await self._conn.commit()
        except Exception as e:
            await self.rollback()
            raise e

    async def rollback(self):
        await self._conn.rollback()

    async def close(self):
        self.pool.close()
        await self.pool.wait_closed()

    def executed(self):
        return self._cur._executed

    async def __aenter__(self):
        self._conn = await self.pool.acquire()
        self._cur = await self._conn.cursor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if not exc_type:
                await self.commit()
            await self._cur.close()
        except Exception as e:
            raise e
        finally:
            await self.pool.release(self._conn)


class PoolWrapper(Pool):
    """Add cache function"""

    def __init__(self, echo, pool_recycle, loop, mincached=0, maxcached=0, minsize=0, maxsize=0, **kwargs):
        kwargs["cursorclass"] = DictCursor
        super(PoolWrapper, self).__init__(minsize, maxsize, echo, pool_recycle, loop, **kwargs)
        if maxcached < mincached:
            raise ValueError("Max cached should be not less than min cached")
        if maxsize < maxcached:
            raise ValueError("Max size should be not less than max cached")
        if minsize < mincached:
            raise ValueError("Min size should be not less than min cached")
        self._mincached = mincached
        self._maxcached = maxcached

    async def _fill_free_pool(self, override_min):
        """Iterate over free connections and remove timeout ones"""
        free_size = len(self._free)
        n = 0
        while n < free_size:
            conn = self._free[-1]
            if conn._reader.at_eof() or conn._reader.exception():
                self._free.pop()
                conn.close()

            elif -1 < self._recycle < self._loop.time() - conn.last_usage:
                self._free.pop()
                conn.close()

            else:
                self._free.rotate()
            n += 1

        while self.size < self.minsize:
            self._acquiring += 1
            try:
                conn = await connect(echo=self._echo, loop=self._loop, **self._conn_kwargs)
                # Raise exception if pool is closing
                self._free.append(conn)
                self._cond.notify()
            finally:
                self._acquiring -= 1
        if self._free:
            return

        if override_min and self.size < self.maxsize:
            self._acquiring += 1
            try:
                conn = await connect(echo=self._echo, loop=self._loop, **self._conn_kwargs)
                # Raise exception if pool is closing
                self._free.append(conn)
                self._cond.notify()
            finally:
                self._acquiring -= 1

    def release(self, conn):
        """
        Release free connection back to the connection pool.
        This is **NOT** a coroutine.
        """
        fut = self._loop.create_future()
        fut.set_result(None)

        if conn in self._terminated:
            assert conn.closed, conn
            self._terminated.remove(conn)
            return fut
        assert conn in self._used, (conn, self._used)
        self._used.remove(conn)
        if not conn.closed:
            in_trans = conn.get_transaction_status()
            if in_trans:
                conn.close()
                return fut
            if self._closing:
                conn.close()
            elif len(self._free) >= self._maxcached:
                conn.close()
            else:
                self._free.append(conn)
            fut = self._loop.create_task(self._wakeup())
        return fut
