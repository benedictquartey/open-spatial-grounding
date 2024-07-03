import numpy as np
import re
from osg.utils.general_utils import parse_spatial_relations

def get_arguments_from_spatial_constraint(spatial_constraint):
    '''
    A function that extracts the arguments from a spatial constraint

    spatial_constraint: a string representing a spatial constraint
    '''
    matches = re.findall('\(([^()]*(?:\([^()]*\)[^()]*)*)\)', spatial_constraint)
    arguments = matches[0].split(",")
    return arguments

def check_spatial_predicate_satisfaction(referent_of_interest, spatial_constaint, all_elements_details):
    element_id = referent_of_interest['mask_id']
    element_position = referent_of_interest['worldframe_3d_position']

    comparative_referent_labels = []
    constraint_args = get_arguments_from_spatial_constraint(spatial_constaint)
    comparative_referent_labels.extend(constraint_args)

    print(f"    Checking Descriptor: {spatial_constaint}  || Comparative referents: {comparative_referent_labels}")
    
    constraint_satisfaction = False
    satisfing_instances = []

    if "isbetween" in spatial_constaint:
        ref1_label,ref2_label = comparative_referent_labels[0],comparative_referent_labels[1]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)
        all_comparison_ref2_elements = all_element_with_label(ref2_label, all_elements_details)

        #check if referent of interest is between any combination of occurances/instance of the two given comparison referents || we need all occurances for optimal planning []
        for ref1_instance in all_comparison_ref1_elements:
            for ref2_instance in all_comparison_ref2_elements:
                satisfaction_value = is_between(element_position, ref1_instance['worldframe_3d_position'], ref2_instance['worldframe_3d_position'])
                # print(f"        {element_id} satisfies {spatial_constaint} between {ref1_instance['mask_id']} and {ref2_instance['mask_id']}: {satisfaction_value}")
                print(f"        '{element_id}' is between '{ref1_instance['mask_id']}' and '{ref2_instance['mask_id']}': {satisfaction_value}")
                if satisfaction_value == True:
                    satisfing_instances.append([ref1_instance['mask_id'],ref2_instance['mask_id']])# record all specific instances of the comparison referents that satisfies the constraint
                    constraint_satisfaction = True #record constraint to be satisfied if any instance of the comparison referents satisfies the constraint
        
    elif "isabove" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is left of any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_above(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is above '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 

    elif "isbelow" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is left of any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_below(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is below '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 

    elif "isleftof" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is left of any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_left_of(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is left of '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 

    elif "isrightof" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is right of any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_right_of(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is right of '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 
    
    elif "isnextto" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is is_next_to any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_next_to(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is next to '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 
   
    elif "isinfrontof" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is is_next_to any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_in_front_of(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is in front of '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 

    elif "isbehind" in spatial_constaint:
        ref1_label = comparative_referent_labels[0]
        all_comparison_ref1_elements = all_element_with_label(ref1_label, all_elements_details)

        #check if referent of interest is is_next_to any occurances/instance of the given comparison referent
        for ref1_instance in all_comparison_ref1_elements:
            satisfaction_value = is_behind(element_position, ref1_instance['worldframe_3d_position'])
            # print(f"        {element_id} satisfies {spatial_constaint} for {ref1_instance['mask_id']}: {satisfaction_value}")
            print(f"        '{element_id}' is behind '{ref1_instance['mask_id']}': {satisfaction_value}")
            if satisfaction_value == True:
                satisfing_instances.append(ref1_instance['mask_id'])
                constraint_satisfaction = True 

    else:
        raise Exception(f"Unrecognized spatial constraint: {spatial_constaint}")
    
    return constraint_satisfaction, satisfing_instances

def all_element_with_label(label, all_elements_details):
    '''
    A function that returns all elements with a given label

    label: string representing the label of the elements we want to return
    all_elements_details: a dictionary containing all the elements in the scene with various details
    '''
    compiled_elements = []
    #if nested descriptor in label only return list of satisfying elements
    if '::' in label:
        actual_label,descriptor = label.split('::')
        print(f"    Comparative referent '{actual_label}' has descriptor '{descriptor}'")
        print(f"    -------------------------------------------------\n    Finding satisfying instances of nested descriptor\n    -------------------------------------------------")
        all_label_elements = [element for element in all_elements_details if element['mask_label'] == actual_label]
        # print(f"found elements ",[element['mask_id'] for element in all_label_elements])

        referent_spatial_details = parse_spatial_relations([label])
        print("    Parsed constraints: ",referent_spatial_details)
        for element in all_label_elements:
            element_spatial_details = referent_spatial_details[element['mask_label']]
            for spatial_constraint in element_spatial_details:
                spatial_satisfaction, satisfing_instances = check_spatial_predicate_satisfaction(element,spatial_constraint,all_elements_details)
                if spatial_satisfaction == True:
                    compiled_elements.append(element)
        print("    Compiled elements: ",[element['mask_id'] for element in compiled_elements])
        print(f"    -------------------------------------------------\n    Resuming Original Grounding\n    -------------------------------------------------")
        return compiled_elements
    else:
        #return list of all elements with the given label
        return [element for element in all_elements_details if element['mask_label'] == label]


########################  GLOBAL FRAME SPATIAL PREDICATES || Spatial reasoning is with respect to worldframe origin  ########################
def is_between(referent_1_position, referent_2_position, referent_3_position, threshold=1.5):
    """
    Determines if the position of referent_1 is approximately between the positions of referent_2 and referent_3.

    This function checks whether referent_1 lies between referent_2 and referent_3, within a specified proximity 
    threshold to the line segment joining referent_2 and referent_3. The function first verifies if referent_1 
    is in the general direction between the other two points, then it checks whether referent_1 is close enough 
    to the line segment formed by referent_2 and referent_3. 

    Parameters:
    referent_1_position (array-like): The 3D position of the first referent to test.
    referent_2_position (array-like): The 3D position of the second referent, one end of the line segment.
    referent_3_position (array-like): The 3D position of the third referent, the other end of the line segment.
    threshold (float, optional): The maximum permissible distance from the line segment for referent_1 to be 
                                 considered 'between'. Defaults to 1.

    Returns:
    bool: True if referent_1_position is approximately between referent_2_position and referent_3_position 
          within the threshold, False otherwise.

    Note:
    The function uses the dot product to verify the general direction and the cross product to check the proximity 
    to the line segment. If referent_1 is in the direction between referent_2 and referent_3 but further away 
    than the threshold from the line segment, it will be considered not between.
    """
    v1 = np.array(referent_2_position) - np.array(referent_1_position)  # vector from referent_1_position to referent_2_position
    v2 = np.array(referent_3_position) - np.array(referent_1_position)  # vector from referent_1_position to referent_3_position
    
    if np.dot(v1, v2) > 0:
        # referent_1_position is not between referent_2_position and referent_3_position if dot product is positive
        return False
    
    # check if referent_1_position is near the line formed by referent_2_position and referent_3_position
    v = np.array(referent_3_position) - np.array(referent_2_position)  # vector from referent_2_position to referent_3_position
    vp = np.array(referent_1_position) - np.array(referent_2_position)  # vector from referent_2_position to referent_1_position
    
    # Cross product between vector v and vp gives a vector perpendicular to the plane formed by v and vp
    cross_product = np.cross(v, vp)

    # If the length of the cross product is small, the point is close to the line
    if np.linalg.norm(cross_product) <= threshold:
        return True
    return False

def is_above(referent_1_position, referent_2_position, threshold=0.1):
    """
    Determines if referent_1 is positioned above referent_2, considering a vertical threshold.

    This function checks if the z-coordinate (height) of referent_1 is greater than the 
    z-coordinate of referent_2 by at least the specified threshold value.

    Parameters:
    referent_1_position (array-like): The position of the first referent in 3D space.
    referent_2_position (array-like): The position of the second referent in 3D space.
    threshold (float, optional): The minimum vertical distance to determine 'above' relation. 
                                 Defaults to 0.1.

    Returns:
    bool: True if referent_1 is above referent_2 by at least the threshold, False otherwise.
    """
    return referent_1_position[2] > referent_2_position[2] + threshold

def is_below(referent_1_position, referent_2_position, threshold=0.1):
    """
    Determines if referent_1 is positioned below referent_2, considering a vertical threshold.

    This function checks if the z-coordinate (height) of referent_1 is less than the 
    z-coordinate of referent_2 by more than the specified threshold value.

    Parameters:
    referent_1_position (array-like): The position of the first referent in 3D space.
    referent_2_position (array-like): The position of the second referent in 3D space.
    threshold (float, optional): The minimum vertical distance to determine 'below' relation. 
                                 Defaults to 0.1.

    Returns:
    bool: True if referent_1 is below referent_2 by more than the threshold, False otherwise.
    """
    return (referent_1_position[2] < referent_2_position[2] - threshold)

def is_left_of(referent_1_position, referent_2_position, threshold=0.1):
    """
    Determines if referent_1 is to the left of referent_2, considering a lateral threshold.

    This function checks if the y-coordinate of referent_1 is greater than the y-coordinate 
    of referent_2 by at least the specified threshold value, indicating leftness in a 
    global frame based on the Y-axis.

    Parameters:
    referent_1_position (array-like): The position of the first referent in 3D space.
    referent_2_position (array-like): The position of the second referent in 3D space.
    threshold (float, optional): The minimum lateral distance to determine 'left of' relation. 
                                 Defaults to 0.1.

    Returns:
    bool: True if referent_1 is left of referent_2 by at least the threshold, False otherwise.
    """
    return (referent_1_position[1] > referent_2_position[1] + threshold) # Global frame based leftness and rightness in respect to the Y-axis ###

def is_right_of(referent_1_position, referent_2_position, threshold=0.1):
    """
    Determines if referent_1 is to the right of referent_2, considering a lateral threshold.

    This function checks if the y-coordinate of referent_1 is less than the y-coordinate 
    of referent_2 by more than the specified threshold value, indicating rightness in a 
    global frame based on the Y-axis.

    Parameters:
    referent_1_position (array-like): The position of the first referent in 3D space.
    referent_2_position (array-like): The position of the second referent in 3D space.
    threshold (float, optional): The minimum lateral distance to determine 'right of' relation. 
                                 Defaults to 0.1.

    Returns:
    bool: True if referent_1 is right of referent_2 by more than the threshold, False otherwise.
    """
    return (referent_1_position[1] < referent_2_position[1] - threshold)  # Global frame based leftness and rightness in respect to the Y-axis ###

def is_next_to(referent_1_position, referent_2_position, threshold=1):
    """
    Determines if two referents are next to each other within a specified threshold.

    This function calculates the Euclidean distance between two referents and checks 
    if this distance is less than the given threshold. If the distance is less than 
    the threshold, it implies that the referents are considered to be next to each other.

    Parameters:
    referent_1_position (array-like): The position of the first referent in 3D space.
    referent_2_position (array-like): The position of the second referent in 3D space.
    threshold (float, optional): The maximum permisible distance from referent_2_position for referent_1 to be considered
                                 'next to' referent_2 Defaults to 0.1.

    Returns:
    bool: True if the referents are next to each other within the threshold, False otherwise.
    """

    return (np.linalg.norm(np.array(referent_2_position)-np.array(referent_1_position)) < threshold) 

def is_in_front_of(referent_1_position, referent_2_position,threshold=0.1):
    """
    Determines if referent_1 is in front of referent_2 in global origin coordinate frame.

    Parameters:
    referent_1_position (array-like): The global position of the first referent.
    referent_2_position (array-like): The global position of the second referent.
    threshold (float, optional): The minimum distance threshold to determine 'in front of' relation. Defaults to 0.1.

    Returns:
    bool: True if referent_1 is in front of referent_2 in the object's local frame, False otherwise.
    """

    return referent_1_position[0] < referent_2_position[0] - threshold    # Global frame based infront and behind in respect to the X-axis ###

def is_behind(referent_1_position, referent_2_position, threshold=0.1):
    """
    Determines if referent_1 is behind referent_2 in global origin coordinate frame.

    Parameters:
    referent_1_position (array-like): The global position of the first referent.
    referent_2_position (array-like): The global position of the second referent.
    threshold (float, optional): The minimum distance threshold to determine 'behind' relation. Defaults to 0.1.

    Returns:
    bool: True if referent_1 is behind referent_2 in the object's local frame, False otherwise.
    """

    return referent_1_position[0] > referent_2_position[0] + threshold   # Global frame based infront and behind in respect to the X-axis ###