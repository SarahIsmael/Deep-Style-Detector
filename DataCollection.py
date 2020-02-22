'''
Created on 25 Mar 2019

@author: sara

This program collect  maximum 20 pages of medieval manuscripts  in latin langauge 
'''

import os, sys
import shutil #for file operations
from sickle import Sickle #API for accessing metadata; install with: !pip install sickle
from urllib import request #to download files
import xml.etree.ElementTree as ET

def get_style(xml_file, style_to_include=['carol', 'textualis', 'gotica', 'textura', 'spaetcar', 'bastarda', 'cursiva']):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    #get prefix of root value to find handNote element
    root_name=root.attrib['{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'].split(' ')[0]
    
    scripts_dict = {}
    for handNote in root.iter('{'+root_name+'}'+'handNote'):
        #skip undesired styles
        try:      
            script=handNote.attrib['script']
        except KeyError:
            script='other'
        
        if script in style_to_include:
            pages = []
            for e in handNote.iter('{'+root_name+'}'+'locus'):
                try:
                    pages.append(e.attrib['from']+'-'+e.attrib['to'])
                except KeyError:
                    pages.append(e.attrib['from'])   
                
            try:
                scripts_dict[script].extend(pages)
            except KeyError:
                scripts_dict[script] = pages
    
    return scripts_dict

def download_images(scripts_dict,record_id, output_dir, maximum_number_images_to_download = 20):
    """
    For a given record, download images for the styles in scripts_dict
    """
    script_images = {}
    record_code = record_id.split('/')[-2]
    document_number = record_code + '-' + record_id.split('/')[-1] 
    page_numbers_from_to = []
    for key in scripts_dict.keys():
        page_numbers=scripts_dict[key]
        script_images[key] = []
        #only download images for scripts that have from
        if len(page_numbers)>0:
            for page_number in page_numbers:
                #from and to are specified
                if ('-' in page_number):
                    page_number_split = page_number.split('-')
                    try:
                        start=int(page_number_split[0].strip('r').strip('v').strip('a'))
                        end=int(page_number_split[1].strip('r').strip('v').strip('a'))
                        page_numbers_from_to.append(start)
                        page_numbers_from_to.append(end)
                    except ValueError:
                        continue
                    #for the first and the last page only get the specified side (exact page number and side)
                    if len(script_images[key])<maximum_number_images_to_download:
                        files = download_image(document_number, page_number_split[0], output_dir, key)
                        for f in files:
                            #don't write files to the output list if they are already written or if the max pages are obtained
                            #e.g if 148v makes the list equal to max then skip 148r and other sides
                            if f not in script_images[key] and len(script_images[key])<maximum_number_images_to_download: 
                                script_images[key].append(f)
                    if len(script_images[key])<maximum_number_images_to_download:
                        files = download_image(document_number, page_number_split[1], output_dir, key)
                        for f in files:
                            if f not in script_images[key] and len(script_images[key])<maximum_number_images_to_download:
                                script_images[key].append(f)
                        
                    #for the pages in between start and end, get pages on both sides:
                    for page in range(start+1,end):
                        page_numbers_from_to.append(page)
                        if len(script_images[key])>=maximum_number_images_to_download:
                            continue
                        files = download_image(document_number, page, output_dir, key)
                        for f in files:
                            if f not in script_images[key] and len(script_images[key])<maximum_number_images_to_download: 
                                script_images[key].append(f)
                        
                else:#if only from is specified
                    try:
                        page_numbers_from_to.append(int(page_number.strip('r').strip('v').strip('a')))
                    except ValueError:
                        page_numbers_from_to.append(page_number)
                    if len(script_images[key])<maximum_number_images_to_download:
                        files = download_image(document_number, page_number, output_dir, key)
                        for f in files:
                            if f not in script_images[key] and len(script_images[key])<maximum_number_images_to_download: 
                                script_images[key].append(f)
                        
    #process scripts that have no from
    for key in scripts_dict.keys():
        if len(page_numbers)==0:
            start = 10
            end = 250
            for page in range(start,end+1):
                print("trying to get page num: ", page, "current number of pages for: ", key, "=", len(script_images[key]))
                if len(script_images[key])>=maximum_number_images_to_download:
                    break
                #check if page is not part of any other style
                if page not in page_numbers_from_to:
                    files = download_image(document_number, page, output_dir, key)
                    for f in files:
                        if f not in script_images[key] and len(script_images[key])<maximum_number_images_to_download: 
                            script_images[key].append(f)
    
    return script_images
    
def download_image(document_number, page_number, output_dir, script): 
    files = []
    try:
        page_number = '{:0>3d}'.format(int(page_number))
    except ValueError:
        try:
            page_number = '{:0>3d}'.format(int(page_number[0:-1])) + page_number[-1]
        except ValueError:
            return files
    
    page_sides = ['v', 'r', '']
    if(page_number[-1] in page_sides):
        page_id = '{d}_{p}'.format(d=document_number, p=page_number)#0019_015r
        image_name = output_dir + script + '-' + page_id + '.jpg'         
        if not os.path.exists(image_name):
            #try to download the file
            download_path = 'https://www.e-codices.ch/en/download/{}/max'.format(page_id)
            downloaded_file = request.urlretrieve(download_path)[0]
            if os.path.isfile(downloaded_file):
                if os.stat(downloaded_file).st_size>1000:
                    os.system('mv {} {}'.format(downloaded_file, image_name))
                else:
                    os.remove(downloaded_file)
        if os.path.isfile(image_name):
            files.append(image_name)
    else:
        for page_side in page_sides:
            page_id = '{d}_{p}{s}'.format(d=document_number, p=page_number, s=page_side)#0019_015r
            image_name = output_dir + script + '-' + page_id + '.jpg'
            if os.path.exists(image_name):
                print("already exists:", image_name)
            else:
                #otherwise try to download the file
                download_path = 'https://www.e-codices.ch/en/download/{}/max'.format(page_id)
                downloaded_file = request.urlretrieve(download_path)[0]
                if os.path.isfile(downloaded_file):
                    if os.stat(downloaded_file).st_size>1000:
                        os.system('mv {} {}'.format(downloaded_file, image_name))
                    else:
                        os.remove(downloaded_file)
            if os.path.isfile(image_name):
                files.append(image_name)
    return files
    
def get_value(meta_data, key):
    encoding_str = 'utf-8'
    try:
        return str(meta_data[key][0].encode(encoding_str)).strip("b").strip("'")
    except KeyError:
        return 'NA'

def get_records(lang = 'lat', tsv_output_file = "dataset_train_info.tsv", 
                image_dir = 'train_images_1Apr/', xml_dir='xmls/', reset_info_file=False):
    #create an object
    sickle = Sickle('https://www.e-codices.unifr.ch/oai/oai.php')
    encoding_str = 'utf-8'
    #get the list of records
    records = sickle.ListRecords(metadataPrefix='oai_dc')
    
    if not os.path.isdir(xml_dir):
        os.makedirs(xml_dir)
    
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    
    with open(tsv_output_file, 'a+') as dataset_file:
        records_downloaded = []
        dataset_file.seek(0)
        for l in dataset_file.readlines():
            records_downloaded.append(l.strip().split('\t')[0])
        
        for n, record in enumerate(records):
            record_id = record.header.identifier
            if record_id in records_downloaded and not reset_info_file:
                continue
            
            print(n,record_id)
            meta_data = record.metadata
            #get values from the meta data dictionary
            lang=get_value(meta_data, 'language')
            date = get_value(meta_data, 'date')
            rights = get_value(meta_data, 'rights')
            
            # This condition will return all images of some type of styles
            if lang==str(lang.encode(encoding_str)).strip("b").strip("'"):
                document_num=record_id.split('/')[-1]
                document_type=record_id.split('/')[-2]
                xml_file_name = document_type+'-'+document_num+'.xml'
                xml_file_path = xml_dir + xml_file_name
                xml_record="https://www.e-codices.ch/xml/tei_published/" + xml_file_name
                #do not download an xml file for the manuscript if it exists
                if not os.path.isfile(xml_file_path):
                    try:
                        downloaded_file = request.urlretrieve(xml_record)[0]
                        shutil.move(downloaded_file, xml_file_path)
                        script_dict=get_style(xml_file_path)
                    except:
                        script_dict = {}
                else:
                    script_dict=get_style(xml_file_path)
                if script_dict!={}:
                    images_scripts_dict = download_images(script_dict,record_id, image_dir)
                    images = ';'.join([x+":"+','.join(y.split('/')[-1] for y in images_scripts_dict[x]) for x in images_scripts_dict.keys()])
                    script = ';'.join([x+":"+','.join(script_dict[x]) for x in script_dict.keys()])
                    line=record_id+'\t'+date+'\t'+rights+'\t'+ script+ '\t' + images + '\n'
                    dataset_file.write(line)
                    print(line)
                
if __name__ == "__main__":
    
    get_records()
    
    