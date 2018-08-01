#Utility function for adding an element to the tag dictionary. The new element should be a 1-D dictionary
def add_element_to_dict(dict, key, new_sub_dict):
	dict[key] = new_sub_dict
	return dict