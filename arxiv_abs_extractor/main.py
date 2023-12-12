from data_extractor import make_dataset_in_group_of_queries




if __name__ == "__main__":
    topics = set()
    while True:
        topic = str(input("Enter topic to extract data (type 'done' when done with topics): "))
        if topic.lower().strip() == "done":
            break
        else:
            topics.add(topic)
    max_results = int(input("Enter max results to extract: "))
    min_num_words = int(input("Enter minimum number of words required in each search: "))
    data_type = str(input("Please enter the data type for the dataset: "))
    make_dataset_in_group_of_queries(topics, max_results, min_num_words=min_num_words, data_type=data_type)