import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process
import matplotlib.patches as patches
from torchvision.ops import box_convert
from segment_anything import build_sam, SamPredictor
from mobile_sam import sam_model_registry, SamPredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection

from osg.spatial_relationships import check_spatial_predicate_satisfaction
from osg.utils.general_utils import get_center_pixel_depth, get_mask_pixels_depth, get_bounding_box_center_depth, get_bounding_box_pixels_depth, pixel_to_world_frame

# import warnings
# warnings.filterwarnings("ignore")
class vlm_library():
     def __init__(self, vl_model, seg_model="sam",tmp_fldr="./tmp"):
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self.vl_model_name = vl_model
          self.seg_model = None
          self.grounded_elements = []
          self.tmp_fldr = tmp_fldr

          #create tmp_fldr folder if it doesn't exist
          if not os.path.exists(self.tmp_fldr):
               os.makedirs(self.tmp_fldr)

          if vl_model == "owl_vit": #Vision Transformer for Open-World Localization [Simple Open-Vocabulary Object Detection with Vision Transformers]
               self.vl_model, self.vl_processor = self.load_owl_vit()

          elif vl_model == "owl_v2": #Scaling Open-Vocabulary Object Detection
               self.vl_model, self.vl_processor = self.load_owl_v2()
          else:
               raise ValueError("Invalid model name")
          
          print(f"\nVisual language model: {self.vl_model_name}\n-------------------------------------------------")
     
          if seg_model == "sam":
               module_path = os.path.dirname(__file__)
               sam_checkpoint = os.path.join(module_path, 'model_ckpts/sam_vit_h_4b8939.pth')
               self.seg_model = self.load_sam(sam_checkpoint)
               print(f"Segmentation model: SAM\n-------------------------------------------------")

          elif seg_model == "mobile_sam":
               module_path = os.path.dirname(__file__)
               sam_checkpoint = os.path.join(module_path, 'model_ckpts/mobile_sam.pt')
               self.seg_model = self.load_mobile_sam(sam_checkpoint)
               print(f"Segmentation model: Mobile SAM\n-------------------------------------------------")
          else:
               raise ValueError("Invalid segmentation model name")

     def load_owl_vit(self, model_name="google/owlvit-base-patch32"):
          model = OwlViTForObjectDetection.from_pretrained(model_name)
          processor = OwlViTProcessor.from_pretrained(model_name)
          return model, processor

     # def load_owl_v2(self, model_name="google/owlv2-base-patch16-ensemble"):
     def load_owl_v2(self, model_name="google/owlv2-large-patch14-ensemble"):
          model = Owlv2ForObjectDetection.from_pretrained(model_name)
          processor = Owlv2Processor.from_pretrained(model_name)
          return model, processor  
     
     def load_sam(self,ckpt_filenmae):
          sam = build_sam(checkpoint=ckpt_filenmae)
          sam.to(device=self.device)
          sam_predictor = SamPredictor(sam)
          return sam_predictor
     
     def load_mobile_sam(self,ckpt_filenmae):
          model_type = "vit_t"
          mobile_sam = sam_model_registry[model_type](checkpoint=ckpt_filenmae)
          mobile_sam.to(device=self.device)
          mobile_sam.eval()
          predictor = SamPredictor(mobile_sam)
          return predictor
       
     #Prompting SAM with detected boxes for segmentation masks
     def segment(self,image, boxes):
          image = np.asarray(image)
          self.seg_model.set_image(image) #get image embedding and retain copy in model
          sam_embedding = self.seg_model.get_image_embedding() #get image embedding
          boxes_xyxy = boxes #already in xyxy format
          transformed_boxes = self.seg_model.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(self.device )

          masks, scores, logits = self.seg_model.predict_torch( #batch prediction on all boxes
               point_coords = None,
               point_labels = None,
               boxes = transformed_boxes,
               multimask_output = False,
          )
          return masks,sam_embedding
     
     def show_mask(self,mask, image, boxes_details,random_color=True,center_pixel=None):
          if random_color:
               color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
          else:
               color = np.array([30/255, 144/255, 255/255, 0.6])
          h, w = mask.shape[-2:]
          mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
          
          annotated_frame_pil = Image.fromarray(image).convert("RGBA")
          mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

          composite = np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

          #add bounding boxes on composite of mask and src image
          boxes, labels, scores = boxes_details
          annotated, ax = plt.subplots(figsize=(10, 10))
          ax.imshow(composite)
          
          for box, label, score in zip(boxes, labels, scores):
               xywh = box_convert(boxes=box, in_fmt="xyxy", out_fmt="xywh").numpy()
               x1, y1, width, height = xywh[0], xywh[1], xywh[2], xywh[3]
               rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
               ax.add_patch(rect)
               ax.text(x1, y1, f"{label}:{score}", color='red', fontsize=10, bbox=dict(facecolor='white', edgecolor='red'))
          if center_pixel!= None:
               x, y = center_pixel
               ax.text(x, y, 'x', color='red', fontsize=20, ha='center', va='center')

          #save annotated segmentation to tmp
          return Image.fromarray(composite),annotated
     
     def label_observation(self, observation, propositions_to_ground, threshold):
          """
          Given an observation, a list of propositions, and a threshold, this function returns a list corresponding to which propositions are true in the observation.

          Parameters:
               observation (str): The observation to be labeled.
               propositions_to_ground (list): A list of propositions to be grounded in the observation.
               threshold (float): The minimum score for a proposition to be labeled.

          Returns:
               bounds:, labels, scores: The bounding boxes, labels, and scores for the propositions that are true in the observation.
          """
          if self.vl_model_name == "owl_vit" or self.vl_model_name == "owl_v2":
               return self.label_observation_owl_vit(observation, propositions_to_ground, threshold)

     def label_observation_owl_vit(self, observation, propositions_to_ground, score_threshold=0.1):
          inputs = self.vl_processor(text=propositions_to_ground, images=observation, return_tensors="pt")
          outputs = self.vl_model(**inputs)

          # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
          target_sizes = torch.Tensor([observation.size[::-1]])
          # Convert outputs (bounding boxes and class logits) to COCO API
          results = self.vl_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

          # Retrieve predictions for the image for the corresponding text queries
          boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

          detected_propositions = []
          for box, score, label in zip(boxes, scores, labels):
               box = [round(i, 2) for i in box.tolist()]
               if score >= score_threshold:
                    detected_propositions.append({"label":propositions_to_ground[label], "bounds":box, "confidence":round(score.item(), 2)})
                    # print(f"Detected {propositions_to_ground[label]} with confidence {round(score.item(), 3)} at location {box}")

          bounds = [d['bounds'] for d in detected_propositions]
          labels = [d['label'] for d in detected_propositions]
          confidence = [round(d['confidence'],2) for d in detected_propositions]

          return torch.Tensor(bounds),labels,confidence

     def plot_boxes(self, image, boxes, labels, scores, plt_size=10,file_name="tmp.png"):
          """
          Displays an image with bounding boxes labels and scores.

          Parameters:
               image (numpy.ndarray): The image array.
               boxes (list): A list of lists containing the [x1, y1, x2, y2] coordinates for each bounding box.
               labels (list): A list of strings containing the label for each bounding box.
          """
          fig, ax = plt.subplots(figsize=(plt_size, plt_size))
          ax.imshow(image)
          for box, label, score in zip(boxes, labels, scores):
               box = [int(coord) for coord in box]
               x1, y1, x2, y2 = box
               width, height = x2 - x1, y2 - y1
               rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
               ax.add_patch(rect)
               # ax.text(x1, y1, f"{label}:{score}", color='b',fontsize=12)
               ax.text(x1, y1, f"{label}:{score}", color='red', fontsize=10, bbox=dict(facecolor='white', edgecolor='red'))
          # plt.show()
          fig.savefig(f"{self.tmp_fldr}/{file_name}", format="png")
          img = np.asarray(Image.open(f"{self.tmp_fldr}/{file_name}"))
          plt.close(fig)
          return img

     def spatial_grounding(self,observation_graph,referent_spatial_details,visualize=False,relevant_element_details=None,use_segmentation=True, multiprocessing=False, workers=1):
          #Uses VLM to find occurance of referents in observations then backprojects referents to 3D space to find all positions of referents
          #Then uses spatial details to filter out positions that do not satisfy spatial details
          #Returns a list of all possible positions of referents that satisfy spatial details along with labels of referents

          propositions = list(referent_spatial_details.keys())
          print(f"Propositions to ground: {propositions}")

          # detect and ground referent
          if relevant_element_details is None:
               if multiprocessing:
                    relevant_element_details = self.multiprocess_obtain_relevant_element_details(observation_graph,propositions,visualize,use_segmentation,workers)
               else:
                    relevant_element_details = self.obtain_relevant_element_details(observation_graph,propositions,visualize,use_segmentation)         

          #spatial filtering
          filtered_relevant_element_details = self.spatial_filtering(relevant_element_details,referent_spatial_details)
          return filtered_relevant_element_details
     
     def obtain_relevant_element_details(self, observation_graph,propositions,visualize, use_segmentation=True):
          relevant_element_details=[]
          # obs_idx_to_alphabet = {0:"a",1:"b",2:"c",3:"d"}
          print(f"-------------------------------------------------\nObtaining Relevant Masks\n-------------------------------------------------")

          for node_idx in tqdm(observation_graph.nodes):
               print(f"Evaluating Waypoint at Node {node_idx}")
               node_element_details = self.process_node(node_idx, observation_graph,propositions,visualize, use_segmentation)
               relevant_element_details.extend(node_element_details)
               
          # self.grounded_elements = list(set(self.grounded_elements))
          print(f"-------------------------------------------------\nObtained details for relevant elements: {self.grounded_elements}")

          #save relevant element details to tmp file as npy file
          np.save(f"{self.tmp_fldr}/relevant_elements_alldetails.npy", relevant_element_details)

          #save only minimal details of relevant elements needded for pointcloud visualization
          self.minimal_relevant_element_details(relevant_element_details,name="relevant_elements_mindetails")

          return relevant_element_details
       
     def spatial_filtering(self,relevant_element_details,referent_spatial_details):
          print(f"\n*************************************************\nBegin Spatial Grounding\n-------------------------------------------------")
          #apply spatial constraints checks all elements in relevant_element_details to filter out ones that dont satisfy
          for element in relevant_element_details:
               print("")
               element["spatial_constraints"]=[]
               element_id = element['mask_id'] #change to mask_id
               element_label = element['mask_label'] #change to mask_id
               element_spatial_details = referent_spatial_details[element_label]
          
               #check element that have spatial constraints to see if they satisfy each one
               if element_spatial_details !=[]:
                    print(f"Current Element {element_id} || Type: {element_label} || Spatial Details: {element_spatial_details}")
                    element_spatial_checks = {}
                    for spatial_constraint in element_spatial_details:  
                         spatial_satisfaction, satisfing_instances = check_spatial_predicate_satisfaction(element,spatial_constraint,relevant_element_details)
                         if spatial_satisfaction == True:
                              # print(f"  Element {element_id} satisfies spatial details: {spatial_constraint}")
                              element_spatial_checks[spatial_constraint]= True
                         elif spatial_satisfaction == False:
                              # print(f"  Element {element_id} does not satisfy spatial details: {spatial_constraint}")
                              element_spatial_checks[spatial_constraint]= False

                         spatial_updates = {"constraint":spatial_constraint,"satisfies_spatial_constraint":spatial_satisfaction,"satisfying_instances":satisfing_instances}
                         element["spatial_constraints"].append(spatial_updates)

                    print(f"    Element {element_id} || Completed spatial checks: {element_spatial_checks}")
                    if any(element_spatial_checks.values()):
                         element['filter_out']=False      #satisfies spatial constraint so keep it                   
               else:
                    print(f"Element {element_id} has no spatial details")
                    element['filter_out']=False #has no spatial constraint so keep it

          #filter out elements that do not satisfy spatial details
          filtered_relevant_element_details = [element for element in relevant_element_details if element.get('filter_out')==False]

          #save only minimal details of filtered_relevant_element_details
          self.minimal_relevant_element_details(filtered_relevant_element_details,name="filtered_relevant_elements_mindetails")

          return filtered_relevant_element_details #sanity test 
     
     def minimal_relevant_element_details(self,relevant_element_details,name):
          minimal_relevant_element_details=[]
          for details in relevant_element_details:
               minimal_relevant_element_details.append({    "mask_id":details['mask_id'],
                                                            "mask_label":details['mask_label'],
                                                            "mask_center_pixel":details['mask_center_pixel'],
                                                            "mask_depth": details['mask_depth'],
                                                            "mask_all_pixels":details['mask_all_pixels'],
                                                            "mask_all_pixels_depth":details['mask_all_pixels_depth'],
                                                            "origin_nodepose":details['origin_nodepose'],
                                                            "worldframe_3d_position":details['worldframe_3d_position'],
                                                            })
          
          np.save(f"{self.tmp_fldr}/{name}.npy", minimal_relevant_element_details)

    #Split a list into evenly sized chunks
     def chunks(self,l, n):
          return [l[i:i+n] for i in range(0, len(l), n)]

     def multiprocess_obtain_relevant_element_details(self, observation_graph,propositions,visualize, use_segmentation=True, workers=1):
          relevant_element_details=[]
          total = len(observation_graph.nodes)
          chunk_size = total // int(workers)
          slice = self.chunks(list(observation_graph.nodes), chunk_size)
          jobs = []
          print(f"--------------------------------------------------------------------------\n Running: VLM Detections || Datanode count: {total} || workers: {workers} || workers_after_chunking: {len(slice)}\n--------------------------------------------------------------------------\n")
          for job_id, tasks_slice in enumerate(slice):
               j = Process(target=self.process_node_thread, args=(job_id, tasks_slice, observation_graph, propositions, visualize, use_segmentation ))
               jobs.append(j)
               j.start()
          
          #  wait for the processes to complete
          for j in jobs:
               j.join()

          #load all relevant element details from tmp files and combine them
          for i in range(workers):
               relevant_element_details.extend(np.load(f"{self.tmp_fldr}/vlmlogs/worker{i}_elements.npy",allow_pickle=True))

          #save relevant element details to tmp file as npy file
          np.save(f"{self.tmp_fldr}/relevant_elements_alldetails.npy", relevant_element_details)

          #save only minimal details of relevant elements needded for pointcloud visualization
          self.minimal_relevant_element_details(relevant_element_details,name="relevant_elements_mindetails")

          return relevant_element_details

     def process_node_thread(self, job_id, tasks_slice, observation_graph, propositions, visualize, use_segmentation):
          worker_elements = []
          exp_dir = os.path.join(self.tmp_fldr+"/vlmlogs")
          try: 
               if not os.path.exists(exp_dir):
                    os.makedirs(exp_dir)
          except:
                    print("Output directory already exists, moving on ...")
          sys.stdout = open(f'{exp_dir}/worker{job_id}.out', 'w+')
          time_init = time.time()

          for i,node_idx in enumerate(tasks_slice):
               print(f"Worker {job_id} || Task {i+1}/{len(tasks_slice)} || Node {node_idx}")
               node_element_details = self.process_node(node_idx, observation_graph, propositions, visualize, use_segmentation)
               worker_elements.extend(node_element_details)
               print(f"Worker {job_id} || Task {i+1}/{len(tasks_slice)} || Node {node_idx} || Completed")
          
          np.save(f"{exp_dir}/worker{job_id}_elements.npy", worker_elements)

          time_end = time.time()
          print(f"Worker {job_id} || Time taken: {time_end-time_init} seconds")
          sys.stdout.close()

     def process_node(self, node_idx ,observation_graph,propositions,visualize, use_segmentation=True): 
          node_element_details = []   
          obs_idx_to_alphabet = {0:"a",1:"b",2:"c",3:"d"}
          print(f"Evaluating Waypoint at Node {node_idx}")
          observation_graph.nodes[node_idx]['annotated_img']={}
           #label for each picture at that node (left, right, front and back)
          for obs_idx in range(4):
               print(f"   Observation_{obs_idx}...")
               bounds,labels,confidence = self.label_observation(observation_graph.nodes[node_idx]['rgb'][obs_idx],propositions,threshold=0.1)

               present_propositions = list(set(labels))
               tracking_ids=[]
               # print(f"Elements in observation: {present_propositions}")
          
               #Check if detected labels/propositions at observation node havent already been segmented and masks obtained 
               if not all(element in self.grounded_elements for element in present_propositions):
                    if use_segmentation:
                         print(f"      Detected Task relevant elements: {present_propositions} || Segmenting to obtain masks ...")
                         # segment for masks of each proposition
                         masks,sam_embedding=self.segment(observation_graph.nodes[node_idx]['rgb'][obs_idx],bounds)
                    else:
                         print(f"      Detected Task relevant elements: {present_propositions} || Using bounding boxes as masks ...")
                         masks = bounds
                         sam_embedding=None

                    #get depth info
                    try :
                         depth_data = observation_graph.nodes[node_idx]['depth_data'][obs_idx]
                         print(f"      Loaded depth data from waypoint node {node_idx}, observation [{obs_idx}]")
                    except:
                         print(f"      No existing depth data || Estimating depth image...")
                         depth_img,depth_data = self.estimate_depth(observation_graph.nodes[node_idx]['rgb'][obs_idx])

                    #get mask info
                    for i,mask in enumerate(masks):
                         print(f"      Processing {labels[i]} mask")
                         if use_segmentation:
                              actual_mask = mask[0].cpu()
                              center_pixel, center_pixel_depth = get_center_pixel_depth(actual_mask,depth_data)
                              mask_pixel_coords,pixel_depths,average_depth=get_mask_pixels_depth(actual_mask,depth_data)
                         else:
                              actual_mask = mask
                              center_pixel, center_pixel_depth = get_bounding_box_center_depth(actual_mask,depth_data)
                              mask_pixel_coords,pixel_depths,average_depth=get_bounding_box_pixels_depth(actual_mask,depth_data)
                         
                         print(f"         Mask {labels[i]}_{str(node_idx)}{obs_idx_to_alphabet[obs_idx]}_{i} || Original Center pixel: {center_pixel} || Center pixel depth: {center_pixel_depth}")
                         center_pixel_depth=average_depth #Use average of mask depth with actual values as depth of center pixel || not just the actual center pixel depth
                         
                         ##Three step adaptive depth data approach 
                         mask_depth=0.0
                         if center_pixel_depth != 0.0:
                              mask_depth=center_pixel_depth
                         if center_pixel_depth == 0.0:
                              successful_pixel = False
                              #Try to get closest pixel to center pixel in mask, that has a depth
                              # Create a list of tuples with pixel indices, depth, and distance from center pixel
                              pixel_data = [(idx, depth, ((coord[0] - center_pixel[0])**2 + (coord[1] - center_pixel[1])**2)**0.5)
                                             for idx, (depth, coord) in enumerate(zip(pixel_depths, mask_pixel_coords))]

                              # Sort the pixel data list based on the distance from center pixel
                              sorted_pixel_data = sorted(pixel_data, key=lambda x: x[2])

                              # Iterate over the sorted pixel data list
                              for idx, depth, distance in sorted_pixel_data:
                                   if depth != 0.0:
                                        mask_depth = depth
                                        center_pixel = mask_pixel_coords[idx]
                                        successful_pixel = True
                                        print(f"         Center pixel depth empty,obtained new pixel from mask || pixel:{center_pixel}, depth: {mask_depth}")
                                        break

                              # Skip monocular model and just move on to next mask if no sensor depth 
                              if successful_pixel == False:
                                   print("         Not using depth model, moving on to next mask")
                                   continue

                         print(f"         Mask {labels[i]}_{str(node_idx)}{obs_idx_to_alphabet[obs_idx]}_{i} || Chosen Center pixel: {center_pixel} || Average mask depth: {average_depth} || Chosen Mask depth: {mask_depth}")

                         mask_label =  labels[i]
                         tracking_id = labels[i]+"_"+str(node_idx)+obs_idx_to_alphabet[obs_idx]+"_"+str(i)
                         tracking_ids.append(tracking_id)
                         if tracking_id not in self.grounded_elements: #only record new masks
                              #get worldframe backprojected 3d position of object(center pixel)
                              print(f"         Backprojectig 3D ray using pixel: {center_pixel} & depth: {mask_depth}m for {tracking_id}...")
                              center_y, center_x = center_pixel
                              pixel_depth = mask_depth
                              rotation_matrix = observation_graph.nodes[node_idx]['pose'][obs_idx]['rotation_matrix']
                              position = observation_graph.nodes[node_idx]['pose'][obs_idx]['position']
                              transformed_point,bad_point = pixel_to_world_frame(center_y,center_x,pixel_depth,rotation_matrix,position)
                              print(f"         Recording mask info for {tracking_id}...")
                              node_element_details.append({"mask_label":mask_label,
                                                       "mask_id":tracking_id,
                                                       "origin_obsnode":node_idx,
                                                       "mask":actual_mask.cpu() if torch.is_tensor(actual_mask) else actual_mask,
                                                       "mask_center_pixel":center_pixel,
                                                       "mask_center_pixel_depth":center_pixel_depth,
                                                       "mask_all_pixels":mask_pixel_coords,
                                                       "mask_all_pixels_depth":pixel_depths,
                                                       "mask_depth":mask_depth,
                                                       "sam_embedding":sam_embedding.cpu() if torch.is_tensor(sam_embedding) else sam_embedding,
                                                       "origin_nodeimg":observation_graph.nodes[node_idx]['rgb'][obs_idx],
                                                       "origin_nodedepthimg":depth_data,
                                                       "origin_nodepose":observation_graph.nodes[node_idx]['pose'][obs_idx],
                                                       "worldframe_3d_position":transformed_point if bad_point==False else None
                                                       })        
                    self.grounded_elements.extend(tracking_ids)
              
                    if visualize: 
                         # save anotated images to tmp folder
                         file_name = f"observation_{node_idx}_{obs_idx_to_alphabet[obs_idx]}.png"
                         annotated = self.plot_boxes(observation_graph.nodes[node_idx]['rgb'][obs_idx], bounds, labels, confidence,plt_size=8,file_name=file_name) #visualize grounding results
                    
          return node_element_details

 
