
file_directory = "./test/"
num_reducer = int(input('num_reducer: '))
testcase = input('testcase: ')

words = dict()
words_cnt = 0
for i in range(num_reducer):
    file = "TEST" + testcase + "-" + str(i) + ".out"
    print(file)
    with open(file_directory+file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(' ')
            word = tmp[0]
            num = int(tmp[1])
            words[word] = num
            words_cnt += 1

word_bank_cnt = 0
file_directory = "./test/"
a = 0
for item in words:
    if(a < 100):
        print(item)
    else:
        break
    a += 1
    
file = testcase + ".ans"
print(file)
with open(file_directory+file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        word_bank_cnt += 1
        tmp = line.split(' ')
        word = tmp[0]
        num = int(tmp[1])
        if(words[word] != num):
            print('Failed')
            exit()

if(word_bank_cnt != words_cnt):
    print('Failed')
    exit()

print('success')

