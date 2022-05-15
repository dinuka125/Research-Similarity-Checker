
sentence_list= []

def get_sentence_list():
    datafile = open("newfile.txt", "r")
    lines = datafile.readlines()
    for line in lines:
        line = line.capitalize()
        sentence_list.append(line.strip())
    return sentence_list
