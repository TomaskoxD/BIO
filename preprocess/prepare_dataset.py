import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import dask
import io

@dask.delayed
class Preprocessor():
    # @dask.delayed
    def get_binary_mask(self,img):
        """
        Get mask for image.
        Args:
            img: Origin image.
        Returns:
            mask: Mask for image.
        """
        
        if img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        threhold = np.mean(gray_img)/3-5
        _, mask = cv2.threshold(gray_img, max(0,threhold), 1, cv2.THRESH_BINARY)
        nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
        new_mask = (1-mask).astype(np.uint8)
        _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
        _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
        mask = mask + new_mask

        # create kernel for morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))

        # morphological opening operation
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        return mask
    # @dask.delayed    
    def get_center_from_edge_of_mask(self,mask):
        """
        Get center for image.
        Args:
            mask: Mask for image.
        Returns:
            center: Center for image.
        """

        center=[0,0]
        x=mask.sum(axis=1)
        center[0]=np.where(x>x.max()*0.95)[0].mean()
        x=mask.sum(axis=0)
        center[1]=np.where(x>x.max()*0.95)[0].mean()
        return center
    # @dask.delayed    
    def get_radius_from_mask_center(self,mask,center):
        """
        Get radius for image.
        Args:
            mask: Mask for image.
            center: Center for image.
        Returns:
            radius: Radius for image.
        """

        mask=mask.astype(np.uint8)
        kernel_size=max(mask.shape[1]//400*2+1,3)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        mask=cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        index=np.where(mask>0)
        b=np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2) # sqrt((x-x0)^2+(y-y0)^2)
        b_count=np.bincount(np.ceil(b).astype(np.int32))
        try:
            radius=np.where(b_count>b_count.max()*0.995)[0].max()
        except:
            print('error in get_radius_from_mask_center\n\n', center, '\n\n', b_count, '\n\n')
            radius = mask.shape[0] - center[0]
        return radius
    # @dask.delayed    
    def get_circle_by_center_bounding_box(self,shape,center,radius):
        """
        Get circle for image.
        Args:
            shape: Shape for image.
            center: Center for image.
            radius: Radius for image.
        Returns:
            center_mask: Circle for image.
        """
        center_mask=np.zeros(shape=shape).astype('uint8')
        center_tmp=(int(center[0]),int(center[1]))
        center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
        return center_mask
    # @dask.delayed    
    def get_mask(self,img):
        """
        Get mask for image.
        Args:
            img: Origin image.
        Returns:
            tmp_mask: Mask for image.
            bbox: Bounding box for image.
            center: Center of image.
            radius: Radius of image.
        """

        # get gray image 
        if img.ndim == 3:
            gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            gray_img =img.copy()
        else:
            raise 'image dim is not 1 or 3'


        h,w = gray_img.shape
        shape=gray_img.shape[0:2]

        # resize image
        gray_img = cv2.resize(gray_img,(0,0),fx = 0.5,fy = 0.5)

        # normalize image
        tgray_img=cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

        tmp_mask=self.get_binary_mask(tgray_img)

        center=self.get_center_from_edge_of_mask(tmp_mask)
        radius=self.get_radius_from_mask_center(tmp_mask,center)        
        
        #resize back
        center = [center[0]*2,center[1]*2]
        radius = int(radius*2)
        s_h = max(0,int(center[0] - radius))
        s_w = max(0, int(center[1] - radius))
        bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
        tmp_mask=self.get_circle_by_center_bounding_box(shape,center,radius)
        return tmp_mask,bbox,center,radius
    # @dask.delayed    
    def remove_back_area(self,img,bounding_box=None,border=None):
        """
        Remove background area.
        Args:
            img: Origin image.
            bbox: Bounding box for image.
            border: Border for image.
        Returns:
            image: Image without background area.
            border: Border for image.
        """

        image=img
        if border is None:
            border=np.array((bounding_box[0],bounding_box[0]+bounding_box[2],bounding_box[1],bounding_box[1]+bounding_box[3],img.shape[0],img.shape[1]),dtype=np.int32)
        image=image[border[0]:border[1],border[2]:border[3],...]
        return image,border
    # @dask.delayed   
    def supplemental_black_area(self,img,border=None):
        """
        Supplement black area to image to make it square.
        Args:
            img: Origin image.
            border: Border for image.
        Returns:
            image: Image with black area.
        """

        image=img
        if border is None:
            h,v=img.shape[0:2]
            max_l=max(h,v)
            if image.ndim>2:
                image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
            else:
                image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
            border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
        else:
            max_l=border[4]
            if image.ndim>2:
                image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
            else:
                image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        image[border[0]:border[1],border[2]:border[3],...]=img
        return image
    
    def mask_image(self,img,mask):
        img[mask<=0,...]=0
        return img
    # @dask.delayed  
    def get_image_without_background(self, img, path, height_width=(800, 800)):
        """
        Preprocess images.
        Args:
            img: Origin image.
        Returns:
            result_img: Preprocessed image.
            borders: Remove border, supplement mask.
            mask: Mask for preprocessed image.
        
        """

        mask, bounding_box, _, _ = self.get_mask(img)
        # print("masking image...")
        cropped_img = self.mask_image(img, mask)
        cropped_img, border = self.remove_back_area(cropped_img,bounding_box=bounding_box)
        mask, _ = self.remove_back_area(mask,border=border)
        # print("supplementing black area...")
        cropped_img = self.supplemental_black_area(cropped_img)
        cropped_img = cv2.resize(cropped_img, height_width)

        if cropped_img.ndim == 3 and cropped_img.shape[2] == 3:
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, cropped_img)
        # print("done.")
        return True


@dask.delayed
class FileHandler():
    
    def image_open(self, file, index):
        """
        Open image from file.
        Args:
            file: Image file.
            index: Image index.
        Returns:
            Image object.
        """
        try:
            img = file.open(file.namelist()[index])
            img = plt.imread(img)
        except:
            img = file.namelist()[index]
            print('Image',img,'cannot be read...')
        
        return img
        
    def image_read(self,file_path, c=None):
        """
        Read image from file_path.
        Args:
            file_path: Image file path.
            c: Color mode.
        Returns:
            Image array.
        """
        
        if c is None:
            image = cv2.imread(file_path)
        else:
            image = cv2.imread(file_path, c)

        if image is None:
            raise 'Image cannot be read...'

        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def image_write(self,file_path, image):
        """
        Write image to file_path.
        Args:
            file_path: Image file path.
            image: Image array.
        """

        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(file_path)
        cv2.imwrite(file_path, image)
        # io.imsave(file_path, image)
        return file_path
    
    def fold_dir(self,folder):
        """
        Create folder if not exists.
        Args:
            folder: Folder path.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder
