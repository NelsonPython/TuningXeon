
## Data Analytics and Machine Learning Acceleration
  
As a branch of artificial intelligence, machine learning is currently attracting much attention. Machine learning-based analytics are also becoming increasingly popular.  When compared to other analytics, machine learning can help IT staff, data scientists, and business teams in many types of organizations to quickly unleash the power of AI. Furthermore, machine learning offers many new commercial and open-source solutions, providing a vast ecosystem for developers. Developers can choose from a variety of open-source machine learning libraries such as  Scikit-learn*,  Cloudera* and  Spark* MLlib.

### Intel&reg; Distribution for Python*

Intel&reg; Distribution for Python* is a Python development toolkit for artificial intelligence software developers. It can be used to accelerate the computational speed of Python on the Intel&reg; Xeon&reg; Scalable Processor platform. It is available with  Anaconda* and it can also be installed and used with Conda*, PIP*, APT GET, YUM, Docker*, and others. 

Intel&reg; Distribution for Python* features:

- Take advantage of the most popular and fastest growing programming language with underlying instruction sets optimized for Intel® architectures.
- Achieve near-native performance through acceleration of core Python numerical and scientific packages that are built using Intel® Performance Libraries.
- Achieve highly efficient multithreading, vectorization, and memory management, and scale scientific computations efficiently across a cluster.
- Core packages include Numba, NumPy, SciPy, and more.  

Reference and download site: [https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
  
### Intel® Distribution of Modin*
  
Modin* is a drop-in replacement for pandas, enabling data scientists to scale to distributed DataFrame processing without having to change API code. Intel® Distribution of Modin* adds optimizations to further accelerate processing on Intel® hardware.

Using this library, you can:

- Process terabytes of data on a single workstation
- Scale from a single workstation to the cloud using the same code
- Focus more on data analysis and less on learning new APIs  
![](images/modin_arc.png)

Intel® Distribution of Modin* Features:  

- Accelerated DataFrame Processing
  - Speed up the extract, transform, and load (ETL) process for large DataFrames
  - Automatically use all of the processing cores available on your machine 

- Optimized for Intel Hardware
  - Scale to terabytes of data using Intel® Optane™ Persistent Memory on a single data science workstation
  - Analyze large datasets (over one billion rows) using HEAVY.AI* analytics

- Compatible with Existing APIs and Engines
  - Change one line of code to use your existing pandas API calls, no matter the scale. Instead of importing pandas as pd, simply import modin.pandas as pd by using this command: `import modin.pandas as pd`
    ```
    # import pandas as pd
    import modin.pandas as pd  
    df = pd.read_csv("my_dataset.csv")
    ```
  - Use Dask*, Ray, or HEAVY.AI compute engines to distribute data without having to write code
  - Continue to use the rest of your Python ecosystem code, such as NumPy, XGBoost, and scikit-learn*
  - Use the same notebook to scale from your local machine to the cloud
   
Reference and download site: [https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.htm](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.htm) 

### Intel® Extension for Scikit-learn*

Intel® Extension for Scikit-learn* can seamlessly speed up your scikit-learn applications for Intel® CPUs and GPUs across single and multi-node configurations. This extension package dynamically patches scikit-learn estimators while improving performance for your machine learning algorithms.

The top benefits are:

- No up-front cost for learning a new API
- Integration with the Python* ecosystem
- Up to 100x better performance and accuracy than the standard scikit-learn

[Learn More about Patching scikit-learn](https://intel.github.io/scikit-learn-intelex/what-is-patching.html)  

![](images/scikit-learn_sample.png)

Reference and download site: [https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html)

### XGBoost* Optimized for Intel® Architecture

Starting with XGBoost* Version 0.81 and later, Intel® has been directly adding many optimizations to provide superior performance on Intel® CPUs. This well-known, machine-learning package for gradient-boosted decision trees now includes drop-in acceleration for Intel® architectures to significantly speed up model training and improve accuracy for better predictions.

Reference and download site: [https://www.intel.com/content/www/us/en/developer/articles/technical/xgboost-optimized-architecture-getting-started.html](https://www.intel.com/content/www/us/en/developer/articles/technical/xgboost-optimized-architecture-getting-started.html)

### Intel® oneAPI Data Analytics Library (oneDAL)

Intel® oneAPI Data Analytics Library (oneDAL) is a library that helps speed up big data analysis by providing highly optimized algorithmic building blocks for all stages of data analytics (preprocessing, transformation, analysis, modeling, validation, and decision making) in batch, online, and distributed processing modes.

The library optimizes data ingestion along with algorithmic computation to increase throughput and scalability. It includes C++ and Java* APIs and connectors to popular data sources such as Spark* and Hadoop*. Python* wrappers for oneDAL are part of Intel Distribution for Python.

In addition to classic features, oneDAL provides DPC++ SYCL API extensions to the traditional C++ interface and enables GPU usage for some algorithms.

The library is particularly useful for distributed computation. It provides a full set of building blocks for distributed algorithms that are independent from any communication layer. This allows users to construct fast and scalable distributed applications using user-preferable communication means.

![](images/oneDAL_arc.png)

Reference and download site: [https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html)
