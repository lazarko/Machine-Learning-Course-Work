### Assignment Questions

1. When the positioning of all the clusters were on the same level (-0.5) and
    when the spreading variable exceeded 0.3 then no solution was found.
    
2. Radial and polynomial kernel implemented. 

3.
    * B value determines the position of the hyperplane (separator) in the graph. 
    * P parameter of the polynomial kernel affects the solution by fitting 
    the separator more to the data set. Increasing the P parameter may increase the bias and over-fitting of the data.
    * Sigma parameter is same as P, but with the opposite effect. Low sigma increases the over fitting, while high sigma 
        increases 
4. Slack variable C determines the allowance of the alpha (support vectors). It determines the allowance of errors
        for example if C is small it will allow more points to be on each side of the hyperplane.
   
5. The 'soft margin' slack variables should be modified if the data is noisy, while if the data is too complex use different kernel function.
