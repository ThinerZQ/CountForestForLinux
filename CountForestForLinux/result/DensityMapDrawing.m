for i=1:9
    dmapFile=sprintf('vidf1_33_000_f%03d_orig.txt',i);
    originFile=sprintf('vidf1_33_000_f%03d_orig.png',i);
    data = importdata(dmapFile);
    origin=imread(originFile); 
     r = data(1);    
     c = data(2);    
     disp = data(3:end);
     vmin = min(disp);
     vmax = max(disp);
    disp = reshape(disp, [c,r])';
    densitymap = uint8( 255 * ( disp - vmin ) / ( vmax - vmin ) );
    
    subplot(211),imshow(origin);
    subplot(212),imagesc(disp);
    
end