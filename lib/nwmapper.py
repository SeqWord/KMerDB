import math

class Mapper:
    def __init__(self):
        self.weights = {"T":[2,0],"C":[3,1],"A":[0,2],"G":[1,3]}
        self.code = {"X":["T","C","A","G"],"Y":["A","G","T","C"]}
    
    #### PUBLIC METHODS
    def __call__(self,word,x=None,y=None):
        # If the arguments are wlength,x,y - return the corresponding word
        if x and y and type(word)==type(1):
            return self.restore(word,x,y)
                
        wlength = len(word)
        if not wlength:
            return
        word = str.upper(word)
        
        # Calculate the list of [wlength,x,y] for a word with a variable N
        pos = str.find(word,"N")
        if pos > -1:
            result = []
            words = [word]
            while pos > -1:
                for i in range(len(words)):
                    words[i] = words[i][:pos]+"A"+words[i][pos+1:]
                    for l in ("T","G","C"):
                        words.append(words[i][:pos]+l+words[i][pos+1:])
                pos = str.find(words[0],"N")
            for w in words:
                result.append(self.__call__(w))
            return result
        # if word length is an odd number, the first letter in the word is added to the end
        if wlength%2==1:
            word += word[0]
        p = len(word)//2
        word1 = word[:p]
        word2 = word[p:]
        x = self.accumulate_y(word1)
        y = self.accumulate_x(word2)
        return [wlength,x,y]
    
    # Create a list of all possible words of the given length
    def generate(self,wlength):
        words = []
        p = 2**(wlength+wlength%2)
        for x in range(p):
            y = 0
            while y < p:
                if wlength%2==1:
                    t = x%4
                    rY = y - y%4 + abs(t-2-2*(t%2))
                    y = y - y%4 + 4
                else:
                    rY = y
                    y += 1
                words.append([wlength,x+1,rY+1])
        return words
    
    # return list of permutations that differ from the given word from k1 to k2 nucleotides
    # w is either list [wlength,x,y] or a sequence 'ATGC'
    def get_permutations(self,w,k2=1,k1=1):
        wlength,x,y = self.parse(w)
        if wlength < k2:
            k2 = wlength
        permutations = []
        p = (wlength+wlength%2)//2-1
        x_permutations = self.nrange(x,p,k2)
        x_permutations.insert(0,[x])
        y_permutations = self.nrange(y,p,k2)
        y_permutations.insert(0,[y])
        nbX = nbY = None
        board = {}
        for m in range(len(x_permutations)):
            for u in range(len(y_permutations)):
                if m+u < k1 or m+u > k2:
                    continue
                for i in range(len(x_permutations[m])):
                    for j in range(len(y_permutations[u])):
                        rX = x_permutations[m][i]
                        rY = y_permutations[u][j]
                        # correction for odd words
                        if wlength%2==1:
                            t = (rX-1)%4
                            rY = (rY-(rY-1)%4) + abs(t-2-2*(t%2))
                        # check for uniqueness
                        if rX not in board:
                            board[rX] = []
                        if rY not in board[rX]:
                            board[rX].append(rY)
                        else:
                            continue
                        permutations.append([wlength,rX,rY])
        return permutations

    # arguments w1 and w2 are either lists [wlength,x,y] or sequences 'ATGC'
    def compare_words(self,w1,w2):
        return self.count_mismatches(w1,w2)

    def right_subword(self,wlength,x,y,target_wl):
        if target_wl >= wlength:
            return wlength,x,y
        q = (wlength+wlength%2)//2-1
        t = (y-1)//(4**q)
        wlength,x,y = self.right_intermediate(wlength,x,y)
        if wlength%2==0:
            # Add last letter to X - high level X component
            x += abs(t-2-2*(t%2))*(4**q)
            # Add last letter to Y - low level Y component
            y += abs((x-1)%4-2-2*(x-1)%2)
        if target_wl < wlength-1:
            return self.right_subword(wlength-1,x,y,target_wl)
        return wlength-1,x,y
        
    def left_subword(self,wlength,x,y,target_wl):
        if target_wl >= wlength:
            return wlength,x,y
        q = (wlength+wlength%2)//2-1
        t = (x-1)//(4**q)
        wlength,x,y = self.left_intermediate(wlength,x,y)
        if wlength%2==0:
            # Add last letter to Y - low level Y component
            y += abs((x-1)%4-2-2*((x-1)%2))
        else:
            # Remove last letter - low level Y
            y = self.hdecrement(wlength-1,y)
            # Add firts letter to Y - high level Y component
            y += abs(t-2-2*(t%2))*(4**(q-1))
        if target_wl < wlength-1:
            return self.left_subword(wlength-1,x,y,target_wl)
        return wlength-1,x,y
    
    def move_right(self,w,shift=1):
        wlength,x,y = self.parse(w)
        if shift == 0:
            return [wlength,x,y]
        if shift > wlength-1:
            return None
        words_toProcess = [[wlength,x,y]]
        for i in range(shift):
            words = []
            for W,X,Y in words_toProcess:
                words.extend(self.right_shift([W,X,Y]))
            words_toProcess = []
            words_toProcess.extend(self.ridof_redundancy(words))
        return words

    def move_left(self,w,shift=1):
        wlength,x,y = self.parse(w)
        if shift == 0:
            return [wlength,x,y]
        if shift > wlength-1:
            return None
        words_toProcess = [[wlength,x,y]]
        for i in range(shift):
            words = []
            for W,X,Y in words_toProcess:
                words.extend(self.left_shift([W,X,Y]))
            words_toProcess = []
            words_toProcess.extend(self.ridof_redundancy(words))
        return words
    
    # Incrementation of words
    def right_increment(self,w,n):
        words = []
        for word in self.generate(n):
            words.append(self.add(w,word))
        return words
    
    def left_increment(self,w,n):
        words = []
        for word in self.generate(n):
            words.append(self.add(word,w))
        return words
    
    # return list of subwords of the length wl
    def constituents(self,w,wl):
        wlength,x,y = self.parse(w)
        if wl == wlength:
            return [[wlength,x,y]]
        if wl > wlength:
            return None
        words = [self.left_subword(wlength,x,y,wl)]
        for i in range(wl,wlength):
            W,X,Y = words[-1]
            next_subword = self.right_shift([W,X,Y],[self.y_index(wlength,x,y,i)])[0]
            words.append(next_subword)
        return words
    
    # return 1 letter shorter word with 1 deletion
    def deletion(self,wlength,x,y,ind):
        if ind >= wlength:
            return None
        if ind == 0:
            return self.right_subword(wlength,x,y,wlength-1)
        if ind == wlength-1:
            return self.left_subword(wlength,x,y,wlength-1)
        q = (wlength+wlength%2)//2-1
        if wlength%2==0:
            if ind <= (wlength+wlength%2)//2-1:
                t = (y-1)/(4**q)
                # Delete one letter
                x = self.hslice(wlength,x,ind)+self.hdecrement(wlength,x,ind)-1
                # Add last letter to X - high level X component
                x += abs(t-2-2*(t%2))*(4**q)
                
                # Remove first after middle letter - high level Y component
                y -= t*(4**q)
                # Increase levels of Y components
                y = self.hincrement(wlength,y)
            else:
                # Delete one letter
                ind = wlength-ind-1
                y = (y//(4**(ind+1)))*(4**(ind+1))+self.hincrement(wlength,y,ind)
                
            # Add last letter to Y - low level Y component
            y += abs((x-1)%4-2-2*(x-1)%2)
        else:
            if ind <= (wlength+wlength%2)//2-1:
                # Delete one letter
                x = self.hslice(wlength,x,ind)+self.hdecrement(wlength,x,ind)-1
                # Decrease levels of Y components
                y = self.hdecrement(wlength,y)
            else:
                t = (x-1)//(4**q)
                # Remove first before middle letter - high level X component
                x -= t*(4**q)

                # Decrease levels of Y components
                y = self.hdecrement(wlength,y)
                # Delete one letter
                ind = wlength-ind-1
                y = self.hslice(wlength,y,ind)+self.hdecrement(wlength,y,ind)-1
                # Add first letter to Y - high level Y component
                y += abs(t-2-2*(t%2))*(4**(q-1))

        wlength -= 1
        return wlength,x,y
        
    # concatenate several words
    def concatenate(self,words):
        if not words:
            return words
        if len(words)==1:
            return words[0]
        word = words[0]
        for i in range(1,len(words)):
            word = self.add(word,words[i])
        return word
    
    def complement(self,w):
        wlength,x,y = self.parse(w)
        rX = rY = 0
        for p in range((wlength+wlength%2)//2-1,-1,-1):
            step = 4**p
            rX += step*abs(x//step - 2*(1+(x//step)%2))
            rY += step*abs(y//step - 2*(1+(y//step)%2))
            x -= step*(x//step)
            y -= step*(y//step)
        return [wlength,rX,rY]

    def revcomplement(self,w):
        wlength,x,y = self.parse(w)
        if wlength%2==0:
            return wlength,y,x
        else:
            W = wlength+1
            words = self.right_shift([W,y,x])
            for wl,wx,wy in words:
                if abs(wx%4 - wy%4) == 2:
                    return wlength,wx,wy
    
    #### PRIVAT METHODS
    def parse(self,w):
        if type(w)==type('ATGC'):
            wlength,x,y = list(self.__call__(w))
        else:
            wlength,x,y = w
        return [wlength,x,y]
    
    def restore(self,wlength,x,y):
        MAXVAL = 2**(wlength+wlength%2)
        if x > MAXVAL or y > MAXVAL:
            return None
        if wlength%2==1 and abs(x%4 - y%4) != 2:
            return None
        x -= 1
        y -= 1
        if wlength%2==1:
            W = (wlength+1)//2
        else:
            W = wlength//2
        word1 = word2 = ""
        while W:
            p = 4**(W-1)
            i = int(round(x//p))
            j = int(round(y//p))
            word1 = self.code["X"][i] + word1
            word2 += self.code["Y"][j]
            x -= i*p
            y -= j*p
            W -= 1
        if wlength%2==1:
            word2 = word2[:-1]
        return word1+word2
    
    def count_mismatches(self,first,second):
        w,x,y = self.parse(first)
        W,X,Y = self.parse(second)
        mismatches = []
        if W == w:
            mismatches.append(self.substract(x,X)+self.substract(y,Y))
        elif W > w:
            words = self.constituents([W,X,Y],w)
            for word in words:
                d,s = self.count_mismatches(first,word)
                mismatches.append(d)
        else:
            words = self.constituents([w,x,y],W)
            for word in words:
                d,s = self.count_mismatches(word,second)
                mismatches.append(d)
        match = min(mismatches)
        indices = []
        for i in range(len(mismatches)):
            if mismatches[i]==match:
                indices.append(i)
        return match,indices
    
    def substract(self,first,second):
        if first==second:
            return 0
        count = 0
        p = 1
        d = first-second
        while d:
            p = 4.0**int(math.floor(math.log(abs(d),4)))
            k = round(abs(d)//p)
            first -= int(k*p*abs(d)//d)
            count += 1
            d = first-second
        return count

        
    def right_shift(self,w,letters=[0,1,2,3]):
        if type(w)==type('ATGC'):
            wlength,x,y = list(self.__call__(w))
        else:
            wlength,x,y = w
        q = (wlength+wlength%2)//2-1
        t = (y-1)//(4**q)
        W,x,y = self.right_intermediate(wlength+wlength%2,x,y)
        # Add last letter to X - high level X component
        x += abs(t-2-2*(t%2))*(4**q)
        if wlength%2==1:
            y = 16*(y//16)+1
            for i in range(3):
                if  abs(x%4 - y%4) == 2:
                    break
                y += 1
            m = 4
        else:
            m = 1
        words = []
        for i in letters:
            words.append([wlength,x,y+m*i])
        return words
    
    def left_shift(self,w,letters=[0,1,2,3]):
        if type(w)==type('ATGC'):
            wlength,x,y = list(self.__call__(w))
        else:
            wlength,x,y = w
        q = (wlength+wlength%2)//2-1
        t = (x-1)//(4**q)
        words = []
        W,x,y = self.left_intermediate(wlength+wlength%2,x,y)
        # Increase levels of X components and normalize
        x = 4*(self.hincrement(wlength,x)//4)+1
        # Remove last letter
        y = self.hdecrement(wlength,(y-1))
        # Add firts letter to Y - high level Y component
        y += abs(t-2-2*(t%2))*(4**q)
        if wlength%2==1:
            # normalize
            y = 4*(y//4)+1
            for i in letters:
                words.append([wlength,x+i,y+abs(i-2-2*(i%2))])
        else:
            y += 1
            for i in letters:
                words.append([wlength,x+i,y])
        return words

    def accumulate_x(self,word):
        wlength = len(word)
        x = 1
        for i in range(len(word)):
            x += self.get_x(word[i],wlength)
            wlength -= 1
        return x

    def accumulate_y(self,word):
        wlength = len(word)
        y = 1
        for i in range(len(word)-1,-1,-1):
            y += self.get_y(word[i],wlength)
            wlength -= 1
        return y
    
    def get_x(self,letter,wlength):
        k = 4**(wlength-1)
        if letter not in self.weights:
            return
        return self.weights[letter][0]*k

    def get_y(self,letter,wlength):
        k = 4**(wlength-1)
        if letter not in self.weights:
            return
        return self.weights[letter][1]*k
    
    def ridof_redundancy(self,items):
        items.sort()
        for i in range(len(items)-1,0,-1):
            if items[i] == items[i-1]:
                del items[i]
        return items

    def right_intermediate(self,wlength,x,y):
        #GTGA 4,4,5
        # First after middle letter become last befor middle
        # For example, in GTGA -> TG-A* 
        # high level Y component become high leve X component
        q = (wlength+wlength%2)//2-1
        t = (y-1)//(4**q)
        # Remove first after middle letter - high level Y component
        if wlength%2==0:
            y -= t*(4**q)
        # Remove first letter - low level X component
        x -= (x-1)%4
        # Decrease levels of letters in X component
        x = self.hdecrement(wlength,x)
        # and recalculate leveles of Y components
        if wlength%2==0:
            y = self.hincrement(wlength,y)
        else:
            y = self.hdecrement(wlength,y)
        return wlength,x,y
    
    # private method to support other public methods
    def left_intermediate(self,wlength,x,y):
        q = (wlength+wlength%2)//2-1
        t = (x-1)//(4**q)
        # Remove first before middle letter - high level X component
        if wlength%2==1:
            x -= t*(4**q)
        # Remove last letter - low level Y component
        y -= (y-1)%4
        # Double decrease leveles of Y components
        if wlength%2==1:
            y = self.hdecrement(wlength,y)-1
            #y = self.hdecrement(wlength-1,y)
        return wlength,x,y

    def hincrement(self,wlength,val,IND=0):
        H = 1
        if not IND:
            IND = (wlength+wlength%2)//2-1
        for p in range(IND,0,-1):
            d = ((val-1)//(4**(p-1)))
            val -= d*(4**(p-1))
            H += d*(4**p)
        H = H%(2**(wlength+wlength%2))
        return H
        
    # hyperdecrement until the index specified by IND
    def hdecrement(self,wlength,val,IND=0):
        H = 1
        for p in range((wlength+wlength%2)//2-1,IND,-1):
            d = ((val-1)//(4**p))
            val -= d*(4**p)
            H += d*(4**(p-1))
        if H==0:
            H=1
        return H
    
    # hyperslice from IND to zero level
    def hslice(self,wlength,val,IND):
        H = 1
        for p in range((wlength+wlength%2)//2-1,-1,-1):
            d = ((val-1)//(4**p))
            val -= d*(4**p)
            if p < IND:
                H += d*(4**p)
        if H==0:
            H=1
        return H
    
    # return a letter code by y-component
    def y_index(self,wlength,x,y,ind):
        if ind >= wlength:
            return None
        p = (wlength+wlength%2)//2 - ind
        if p <= 0:
            return ((y-1)//4**((wlength+wlength%2)//2-1+p))%4
        else:
            t = ((x-1)//(4**((wlength+wlength%2)//2-p)))%4
            return abs(t-2-2*(t%2))
        
    # concatenate 2 words
    def add(self,first,second):
        if type(first)==type('ATGC'):
            wl1,x,y = list(self.__call__(first))
            first = [wl1,x,y]
        else:
            wl1,x,y = first
        if type(second)==type('ATGC'):
            wl2,x,y = list(self.__call__(second))
            second = [wl2,x,y]
        else:
            wl2,x,y = second
        wlength = wl1+wl2
        indx = (wl2-wl1)//2
        indy = indx+1
        x = y = 1
        for p in range((wlength+wlength%2)//2-1,-1,-1):
            # calculate X
            if indx <= 0:
                W,X,Y = first
                t = self.y_index(W,X,Y,wl1+indx-1)
            else:
                W,X,Y = second
                t = self.y_index(W,X,Y,indx-1)
            x += int(abs(t-2-2*(t%2))*(4**p))
            indx -= 1
            # calculate Y
            if indy <= 0:
                W,X,Y = first
                t = self.y_index(W,X,Y,wl2+indy-1)
            else:
                W,X,Y = second
                t = self.y_index(W,X,Y,indy-1)
            y += int(t*(4**p))
            indy += 1

        return wlength,x,y
    
    # Next two functions are used together with the get_permutations function to 
    # create a list of permutations
    def nrange(self,n,p,k):
        if k==0 or p<0:
            return [[n]]
        dN = 0
        rP = p
        index_before = index_after = 0
        permutations = [[]]
        orders = []
        while p >= 0:
            step = 4**p
            orders.append((n-dN-1)//step)
            first = n - step*orders[-1]
            added_after = 0
            for i in range(4):
                rN = first + step*i
                if rN != n:
                    if rN > n:
                        permutations[-1].insert(len(permutations[-1])-index_after,rN)
                        added_after += 1
                    else:
                        permutations[-1].insert(index_before,rN)
                        index_before += 1
            index_after += added_after
            dN += step*orders[-1]
            p -= 1
        if k>1:
            permutations.extend(self.populate(permutations[-1],orders,rP,k))
        return permutations
            
    def populate(self,seeds,orders,p,k):
        indices = []
        elements = []
        indices.extend(seeds)
        for m in range(len(orders)-1):
            order = orders[m]
            step = 4**p
            n_elements = [0]
            for i in range(3,-1,-1):
                if order==i:
                    continue
                elif i < order:
                    index = 0
                else:
                    index = -1
                shift = i*step
                if not n_elements[0]:
                    n_elements[0] = indices.pop(index)-shift
                n_elements.append(shift)
            n_permutations = self.nrange(n_elements[0],p-m-1,k-1)
            n_permutations[0] = n_permutations[0]*3
            for j in range(len(n_permutations[0])):
                if j < len(n_permutations[0])//3:
                    shift = n_elements[1]
                elif j >= 2*len(n_permutations[0])//3:
                    shift = n_elements[3]
                else:
                    shift = n_elements[2]
                n_permutations[0][j] += shift
            elements.append(n_permutations[0])
            p -= 1
        return elements
    
###################################################
if __name__ == "__main__":
    #import word_db
    pass
    tester = Mapper()
    #print tester.compare_words("AAGGCAAGGATGGACG","AAGGA")
    word = "TCCATCGG"
    wlength,x,y = tester(word)
    print(word,wlength,x,y)
    '''
    permutations = tester.left_increment(word,2)
    print(permutations)
    print(len(permutations))
    oligos = []
    for item in permutations:
        wlength,x,y = item
        oligo = tester(wlength,x,y)
        print(oligo)
        if oligo in oligos:
            5/0
        oligos.append(oligo)
    '''
