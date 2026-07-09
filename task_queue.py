
#队列处理图片
class TaskQueue:


    def __init__(self):
        self.pics = []
#入队
    def enqueue(self, pic):

        self.pics.append(pic)
#出
    def dequeue(self):
        if self.is_empty():
            return None
        return self.pics.pop(0)
#队首
    def front(self):
        if self.is_empty():
            return None
        return self.pics[0]

    def is_empty(self):
        return len(self.pics) == 0

    def size(self):
        return len(self.pics)




