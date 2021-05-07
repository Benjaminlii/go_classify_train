if __name__ == "__main__":
    dir = {'可回收物':'1','厨余垃圾':'2','其他垃圾':'3','有害垃圾':'4'}
    with open("dir_label.txt", 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))

        for i in enumerate(imgs_info):
            arr = i[1]

            name = arr[0]
            id = arr[1]
            parent_id = '0'
            strs = name.split("_")
            if len(strs) > 1:
                parent_id = dir[strs[0]]
                name = strs[1]
            print("insert into go_classify.garbage_types\n"
                  "(id, created_at, updated_at, deleted_at, row, name, parent_type_id, image_id, garbage_detail_id)\n"
                  "values (%s, now(), now(), null, '', \'%s\', %s, null, null);\n"
                  %(id, name, parent_id))
