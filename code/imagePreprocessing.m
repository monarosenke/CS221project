% filter images


%% edge filter
direct = '/Users/mona/work/githubRepos/CS221project/processedImages/';
% pwd to folder with images to filter
S = dir('*.jpg');
name = 'edgeFiltered';
mkdir([direct name])
for ind = 1:length(S)
    I = imread(S(ind).name);
    I = rgb2gray(I);
    normImage = im2double(I);
    ef = edge3(normImage,'approxcanny',0.3);
    ef = double(ef);
    ef = repmat(ef,1,1,3);
    imwrite(ef, [direct, name '/' S(ind).name]);
end

%% gradient filter
direct = '/Users/mona/work/githubRepos/CS221project/processedImages/';
% pwd to folder with images to filter
S = dir('*.jpg');
name = 'gradientFiltered';
mkdir([direct name])
for ind = 1:length(S)
    I = imread(S(ind).name);
    I = rgb2gray(I);
    
    [Gx, Gy, ~] = imgradientxyz(I);
    normImage = im2double(Gx);
    imwrite(normImage, [direct, name '/' S(ind).name]);
end


