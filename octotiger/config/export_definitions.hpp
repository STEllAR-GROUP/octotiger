//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(OCTOTIGER_EXPORT_DEFINITIONS_AUG_25_2017_0653PM)
#define OCTOTIGER_EXPORT_DEFINITIONS_AUG_25_2017_0653PM

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
# define OCTOTIGER_SYMBOL_EXPORT      __declspec(dllexport)
# define OCTOTIGER_SYMBOL_IMPORT      __declspec(dllimport)
# define OCTOTIGER_SYMBOL_INTERNAL    /* empty */
# define OCTOTIGER_APISYMBOL_EXPORT   __declspec(dllexport)
# define OCTOTIGER_APISYMBOL_IMPORT   __declspec(dllimport)
#elif defined(__NVCC__) || defined(__CUDACC__)
# define OCTOTIGER_SYMBOL_EXPORT      /* empty */
# define OCTOTIGER_SYMBOL_IMPORT      /* empty */
# define OCTOTIGER_SYMBOL_INTERNAL    /* empty */
# define OCTOTIGER_APISYMBOL_EXPORT   /* empty */
# define OCTOTIGER_APISYMBOL_IMPORT   /* empty */
#elif defined(OCTOTIGER_HAVE_ELF_HIDDEN_VISIBILITY)
# define OCTOTIGER_SYMBOL_EXPORT      __attribute__((visibility("default")))
# define OCTOTIGER_SYMBOL_IMPORT      __attribute__((visibility("default")))
# define OCTOTIGER_SYMBOL_INTERNAL    __attribute__((visibility("hidden")))
# define OCTOTIGER_APISYMBOL_EXPORT   __attribute__((visibility("default")))
# define OCTOTIGER_APISYMBOL_IMPORT   __attribute__((visibility("default")))
#endif

// make sure we have reasonable defaults
#if !defined(OCTOTIGER_SYMBOL_EXPORT)
# define OCTOTIGER_SYMBOL_EXPORT      /* empty */
#endif
#if !defined(OCTOTIGER_SYMBOL_IMPORT)
# define OCTOTIGER_SYMBOL_IMPORT      /* empty */
#endif
#if !defined(OCTOTIGER_SYMBOL_INTERNAL)
# define OCTOTIGER_SYMBOL_INTERNAL    /* empty */
#endif
#if !defined(OCTOTIGER_APISYMBOL_EXPORT)
# define OCTOTIGER_APISYMBOL_EXPORT   /* empty */
#endif
#if !defined(OCTOTIGER_APISYMBOL_IMPORT)
# define OCTOTIGER_APISYMBOL_IMPORT   /* empty */
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros used by the runtime module
#if defined(OCTOTIGER_EXPORTS)
# define  OCTOTIGER_EXPORT             OCTOTIGER_SYMBOL_EXPORT
# define  OCTOTIGER_EXCEPTION_EXPORT   OCTOTIGER_SYMBOL_EXPORT
# define  OCTOTIGER_API_EXPORT         OCTOTIGER_APISYMBOL_EXPORT
#else
# define  OCTOTIGER_EXPORT             OCTOTIGER_SYMBOL_IMPORT
# define  OCTOTIGER_EXCEPTION_EXPORT   OCTOTIGER_SYMBOL_IMPORT
# define  OCTOTIGER_API_EXPORT         OCTOTIGER_APISYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros to be used for component modules
#if defined(OCTOTIGER_PLUGIN_EXPORTS)
# define  OCTOTIGER_PLUGIN_EXPORT     OCTOTIGER_SYMBOL_EXPORT
#else
# define  OCTOTIGER_PLUGIN_EXPORT     OCTOTIGER_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros to be used for component modules
#if defined(OCTOTIGER_LIBRARY_EXPORTS)
# define  OCTOTIGER_LIBRARY_EXPORT    OCTOTIGER_SYMBOL_EXPORT
#else
# define  OCTOTIGER_LIBRARY_EXPORT    OCTOTIGER_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#if defined(OCTOTIGER_EXPORTS) || defined(OCTOTIGER_PLUGIN_EXPORTS) || \
    defined(OCTOTIGER_APPLICATION_EXPORTS) || defined(OCTOTIGER_LIBRARY_EXPORTS)
# define OCTOTIGER_ALWAYS_EXPORT      OCTOTIGER_SYMBOL_EXPORT
#else
# define OCTOTIGER_ALWAYS_EXPORT      OCTOTIGER_SYMBOL_IMPORT
#endif

#endif
