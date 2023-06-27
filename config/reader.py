from json import loads


def read(filepath):
    with open(filepath, 'r') as file:
        data = loads(
            file.read()
        )

    file.close()
    
    return data
