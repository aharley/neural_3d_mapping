import random

class DoublePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            self.images = []
            
    def fetch(self):
        return self.embeds, self.images
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds, images):
        # embeds is B x ... x C
        # images is B x ... x 3

        for embed, image in zip(embeds, images):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
                self.images.pop(0)
            # add to the back
            self.embeds.append(embed)
            self.images.append(image)
        return self.embeds, self.images

