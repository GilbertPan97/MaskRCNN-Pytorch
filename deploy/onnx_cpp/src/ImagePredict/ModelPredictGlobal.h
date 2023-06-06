#ifndef MODELPREDICTGLOBAL_H
#define MODELPREDICTGLOBAL_H

#ifndef MP_STATIC
#ifdef MP_SHARED_EXPORT
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __declspec(dllimport)
#endif
#else
#define MP_EXPORT
#endif

#endif // MODELPREDICTGLOBAL_H