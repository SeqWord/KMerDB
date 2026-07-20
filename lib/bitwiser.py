import tools

#### string of bits starts with 1 and followed by 3 digets per character in a reverse order:
# A = '010'
# B = '110'
# C = '101'
# string for ABC:
# '1 101 110 010'

# val must be from 0 to 3
def insert(N,ID,val):
    if val > 3:
        val = 3
    elif val < 0:
        val = 0
    n = int()
    n = int(N >> 3*ID)
    delta = N-(n << 3*ID)
    coeff = [0,0,0]
    for i in range(val):
        coeff[i] = 1
    a,b,c = coeff
    return int(((8*n + 4*a + 2*b + c) << 3*ID)+delta)

# val must be "--", "+", "++" and "+++"
def insert_textVal(N,ID,s):
    values = ["--","+","++","+++"]
    return insert(N,ID,values.index(s))

def replace(N,ID,val):
    n = int()
    n = int(N >> 3*ID)
    delta = N-(n << 3*ID)
    n = n >> 3
    coeff = [0,0,0]
    for i in range(val):
        coeff[i] = 1
    a,b,c = coeff
    return int(((8*n + 4*a + 2*b + c) << 3*ID)+delta)

def delete(data,ID):
    n = int()
    n = int(data >> 3*ID)
    delta = data-(n << 3*ID)
    n >>= 3
    return int((n << 3*ID) + delta)

def merge(first,second,ID):
    if ID == 0:
        return second
    second <<= 3*ID
    return second | first

def merge_list(datalist,indices):
    if len(datalist)==1:
        return datalist[0]
    data = 1
    index = 0
    for i in range(len(datalist)):
        data = merge(data,datalist[i],index)
        index += indices[i]
    return data

# calculate averages and covert them to the values 0, 1, 2, 3
def merge_values(datalist):
    data = 1
    for i in range(len(datalist)):
        av,stdev = data_stat(datalist[i])
        data = insert(data,i,int(float(av)/2.5))
    return data

# return i-values from 0 to 3
def get_value(data,ID):
    n = int(data >> 3*ID)
    i = 0
    for j in range(3):
        i += n%2
        n = n >> 1
    return i

# return upper borders of the ranks in the range of 1-11
def get_rankborder(data,ID):
    n = int()
    n = int(data >> 3*ID)
    i = 1
    increments = [2.5,5,2.5]
    for j in range(3):
        if n%2==1:
            i += increments[j]
        n = n >> 1
    return i

# return upper borders of the ranks in the range of .5, 2.5, 7.5, 9.5
def get_rankmedian(data,ID):
    n = int()
    n = int(data >> 3*ID)
    i = .5
    increments = [2,5,2]
    for j in range(3):
        if n%2==1:
            i += increments[j]
        n = n >> 1
    return i

# compare two genomes
def compare(data,ID_1,ID_2):
    return abs(get_rankmedian(data,ID_1)-get_rankmedian(data,ID_2))

# return the size of a dataset and namber of mismatches
# maximal number of mismatches = 3*size
def match(first,second):
    data = first ^ second
    total = 0
    count = 0
    while first != 1:
        for i in range(3):
            if data%2 == 1:
                count += 1
            data >>= 1
        total += 1
        first >>= 3
        second >>= 3
    if first != second:
        raise Error("Error in bitwiser.match")
    return count,total

# return average and std deviation by ranks
# stat is calculated for the list of sequencial IDs from the start ID
def get_wordStatistics(data,start,n):
    values = []
    for i in range(start,start+n):
        values.append(get_rankborder(data,i))
    sum = sum2 = 0
    n = len(values)
    for val in values:
        sum += val
        sum2 += val*val
    return float(sum)/n - 1.0, float(sum2)/n - (float(sum)/n)**2

# return values as "-", "+", "++" and "+++"
def get_textVal(data,ID):
    val = ""
    n = int()
    n = int(data >> 3*ID)
    for i in range(3):
        if n%2==1:
            val += "+"
        n = n >> 1
    if not val:
        val = "--"
    return val

# each value is represented by 3 digets
def subset(N,IDs):
    subset = 1
    n = int()
    for i in range(len(IDs)-1,-1,-1):
        ID = IDs[i]
        n = int(N >> 3*(ID+1))
        delta = (N >> 3*ID) - (n << 3)
        subset = (subset << 3) + delta
    return subset

# frequencies of scarce, common, frequent and abundant words for the given dataset
def parse_data(data):
    stat = {0:0,4:0,6:0,7:0}
    i = 0
    while data != 1:
        try:
            stat[int(data & 7)] += 1
        except:
            print(int(data & 7),i)
            print(data >> 3)
            return i
        data = data >> 3
        i += 1
    return stat[0],stat[4],stat[6],stat[7]

#### STATISTICS
# word average and variance
def data_stat(data):
    dic = {0:0,4:2.5,6:7.5,7:10}
    x = x2 = count = 0
    while data != 1:
        val = dic[int(data & 7)]
        x += val
        x2 += val*val
        data = data >> 3
        count += 1
    return float(x)/count,float(x2)/count - float(x*x)/count/count

# calculate divergency of word distribution in v genomes
def divergency(data,v):
    table = get_table(data,v)
    # max theoretical variance
    if sum(table)%2==0:
        max_sample = [0]*(sum(table)//2) + [1.0]*(sum(table)//2)
    else:
        max_sample = [0]*((sum(table)+1)//2) + [1.0]*((sum(table)-1)//2)

    # Creation of sample from table:
    the_sample = tools.convert_to_sample(table)
    # variance not defined in sample size < 2
    if sum(table)<2:
        return 0
    return 10.0*tools.variance(the_sample)/tools.variance(max_sample)

# calculate commonality coefficient of word distribution for v genomes
def commonality(data,v):
    table = get_table(data,v)
    the_sample = tools.convert_to_sample(table)
    return 10.0*sum(the_sample)/float(len(the_sample))
    
# table of word distribution [--,+,++,+++] in v genomes
def get_table(data,v):
    table = [0,0,0,0]
    for i in range(v):
        table[get_value(data,i)] += 1
    return table

def denaryToString(n):
    output = ""
    if not n:
        return "0"
    while n:
        if n%2 == 1:
            output = "1"+output
        else:
            output = "0"+output
        n //= 2
    return output

# combine_binary_strings(311, 2) returns 1246, which is "100110111" + "10"
def combine_binary_strings(right_part, left_part, left_part_length=2):
    return (right_part << left_part_length) | left_part

###################################################
if __name__ == "__main__":
    print(combine_binary_strings(311, 2))
    #print(int("10011011110",2))
    #print(denaryToString(316))
    #print(get_value(309485009821063593747808255,5419965))
