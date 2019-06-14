# Author: Johan Mgina
#

from threading import Lock

class SequentialItemBuffer:
    def __init__(self, name="image_buffer"):
        # data
        self.seq_item_list = []        # data saved in tuples (sq_nr, image)
        self.list_lock = Lock()
        self.name = name

    def putLast(self, seq_item, timeout=-1):
        if not seq_item:
            return False
        if not (len(seq_item) == 2):
            return False
        if(seq_item[0] == -1):
            return False

        unlocked = self.list_lock.acquire(timeout=timeout)

        if(unlocked):
            self.seq_item_list.append(seq_item)
            self.list_lock.release()

        return unlocked

    def getFirst(self, timeout=-1):
        unlocked = self.list_lock.acquire(timeout=timeout)
        seq_item = (-1, None)
        if(unlocked):
            if(len(self.seq_item_list) > 0):
                seq_item = self.seq_item_list.pop(0)
            else:
                unlocked = False
            self.list_lock.release()

        return unlocked, seq_item

    def getSeqItemUsingNr(self, seq_nr, timeout=33):
        unlocked = self.list_lock.acquire(timeout=timeout)
        seq_item = (-1, None)
        if(unlocked):
            for i, seq_itm in enumerate(self.seq_item_list):
                if(seq_itm[0] == seq_nr):
                    seq_item = self.seq_item_list.pop(i)
                    break
            self.list_lock.release()
            if(seq_item[0] == -1):
                unlocked = False
        return unlocked, seq_item

    def getLength(self, timeout=-1):
        unlocked = self.list_lock.acquire(timeout=timeout)
        length = -1
        if(unlocked):
            length = len(self.seq_item_list)
            self.list_lock.release()
        return unlocked, length

    def sortBuffer(self, timeout=-1):
        unlocked = self.list_lock.acquire(timeout)
        if(unlocked):
            if(len(self.seq_item_list) > 0):
                self.seq_item_list.sort()
            else:
                unlocked = False
            self.list_lock.release()

        return unlocked

    def cutBuffer(self, start_i, end_i):
        unlocked = self.list_lock.acquire(timeout=-1)
        out_buffer = None
        if(unlocked):
            out_buffer = SequentialItemBuffer("cut_"+self.name)
            length = len(self.seq_item_list)
            if(length > 0 and (not start_i < 0) and end_i > 0 and start_i < end_i and start_i < length):
                if(end_i > length):
                    end_i = length
                for i in range(start_i, end_i):
                    out_buffer.putLast(self.seq_item_list.pop(start_i))
            else:
                unlocked = False
            self.list_lock.release()
        return unlocked, out_buffer
