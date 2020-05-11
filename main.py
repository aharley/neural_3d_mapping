from model_carla_static import CARLA_STATIC
# from model_carla_flo import CARLA_FLO
# from model_carla_time import CARLA_TIME
# from model_carla_reloc import CARLA_RELOC
# from model_carla_obj import CARLA_OBJ
# from model_carla_focus import CARLA_FOCUS
# from model_carla_track import CARLA_TRACK
# from model_carla_siamese import CARLA_SIAMESE
# from model_carla_vsiamese import CARLA_VSIAMESE
# from model_carla_rsiamese import CARLA_RSIAMESE
# from model_carla_msiamese import CARLA_MSIAMESE
# from model_carla_ssiamese import CARLA_SSIAMESE
# from model_carla_csiamese import CARLA_CSIAMESE
# from model_carla_genocc import CARLA_GENOCC
# from model_carla_gengray import CARLA_GENGRAY
# from model_carla_vqrgb import CARLA_VQRGB
# from model_carla_vq3drgb import CARLA_VQ3DRGB
# from model_carla_moc import CARLA_MOC
# from model_carla_don import CARLA_DON
# from model_kitti_don import KITTI_DON
# from model_kitti_moc import KITTI_MOC
# from model_carla_zoom import CARLA_ZOOM
# from model_kitti_zoom import KITTI_ZOOM
# from model_kitti_siamese import KITTI_SIAMESE
# from model_carla_ml import CARLA_ML
# from model_carla_bottle import CARLA_BOTTLE
# from model_carla_pretty import CARLA_PRETTY
# from model_carla_bench import CARLA_BENCH
# from model_carla_reject import CARLA_REJECT
# from model_carla_auto import CARLA_AUTO
# from model_carla_ret import CARLA_RET
# from model_clevr_vq3drgb import CLEVR_VQ3DRGB
# from model_clevr_gen3dvq import CLEVR_GEN3DVQ
# from model_carla_gen3dvq import CARLA_GEN3DVQ
# from model_carla_precompute import CARLA_PRECOMPUTE
# from model_carla_propose import CARLA_PROPOSE
# from model_carla_det import CARLA_DET
# from model_intphys_det import INTPHYS_DET
# from model_intphys_forecast import INTPHYS_FORECAST
# from model_carla_forecast import CARLA_FORECAST
# from model_carla_pipe import CARLA_PIPE
# from model_intphys_test import INTPHYS_TEST
# from model_carla_pwc import CARLA_PWC
import hyperparams as hyp
import os
import cProfile
import logging

logger = logging.Logger('catch_all')

def main():
    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)
    
    if hyp.do_carla_static:
        log_dir_ = os.path.join("logs_carla_static", hyp.name)
    elif hyp.do_carla_flo:
        log_dir_ = os.path.join("logs_carla_flo", hyp.name)
    elif hyp.do_carla_time:
        log_dir_ = os.path.join("logs_carla_time", hyp.name)
    elif hyp.do_carla_reloc:
        log_dir_ = os.path.join("logs_carla_reloc", hyp.name)
    elif hyp.do_carla_obj:
        log_dir_ = os.path.join("logs_carla_obj", hyp.name)
        log_dir_ = os.path.join("logs_carla_obj2", hyp.name)
    elif hyp.do_carla_focus:
        log_dir_ = os.path.join("logs_carla_focus", hyp.name)
    elif hyp.do_carla_track:
        log_dir_ = os.path.join("logs_carla_track", hyp.name)
    elif hyp.do_carla_siamese:
        # log_dir_ = os.path.join("logs_carla_siamese", hyp.name)
        log_dir_ = os.path.join("logs_carla_siamese_rob", hyp.name)
    elif hyp.do_carla_vsiamese:
        log_dir_ = os.path.join("logs_carla_vsiamese", hyp.name)
    elif hyp.do_carla_rsiamese:
        log_dir_ = os.path.join("logs_carla_rsiamese", hyp.name)
    elif hyp.do_carla_msiamese:
        log_dir_ = os.path.join("logs_carla_msiamese", hyp.name)
    elif hyp.do_carla_ssiamese:
        log_dir_ = os.path.join("logs_carla_ssiamese", hyp.name)
    elif hyp.do_carla_csiamese:
        log_dir_ = os.path.join("logs_carla_csiamese", hyp.name)
    elif hyp.do_carla_genocc:
        log_dir_ = os.path.join("logs_carla_genocc", hyp.name)
    elif hyp.do_carla_gengray:
        log_dir_ = os.path.join("logs_carla_gengray", hyp.name)
    elif hyp.do_carla_vqrgb:
        log_dir_ = os.path.join("logs_carla_vqrgb", hyp.name)
    elif hyp.do_carla_moc:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_carla_don:
        log_dir_ = os.path.join("logs_carla_don", hyp.name)
    elif hyp.do_kitti_don:
        log_dir_ = os.path.join("logs_kitti_don", hyp.name)
    elif hyp.do_kitti_moc:
        log_dir_ = os.path.join("logs_kitti_moc", hyp.name)
    elif hyp.do_carla_zoom:
        log_dir_ = os.path.join("logs_carla_zoom", hyp.name)
    elif hyp.do_kitti_zoom:
        log_dir_ = os.path.join("logs_kitti_zoom", hyp.name)
    elif hyp.do_kitti_siamese:
        log_dir_ = os.path.join("logs_kitti_siamese", hyp.name)
    elif hyp.do_carla_ml:
        log_dir_ = os.path.join("logs_carla_ml", hyp.name)
    elif hyp.do_carla_bottle:
        log_dir_ = os.path.join("logs_carla_bottle", hyp.name)
    elif hyp.do_carla_pretty:
        log_dir_ = os.path.join("logs_carla_pretty", hyp.name)
    elif hyp.do_carla_bench:
        log_dir_ = os.path.join("logs_carla_bench", hyp.name)
    elif hyp.do_carla_reject:
        log_dir_ = os.path.join("logs_carla_reject", hyp.name)
    elif hyp.do_carla_auto:
        log_dir_ = os.path.join("logs_carla_auto", hyp.name)
    elif hyp.do_carla_ret:
        log_dir_ = os.path.join("logs_carla_ret", hyp.name)
    elif hyp.do_carla_vq3drgb:
        log_dir_ = os.path.join("logs_carla_vq3drgb", hyp.name)
    elif hyp.do_clevr_vq3drgb:
        log_dir_ = os.path.join("logs_clevr_vq3drgb", hyp.name)
    elif hyp.do_clevr_gen3dvq:
        log_dir_ = os.path.join("logs_clevr_gen3dvq", hyp.name)
    elif hyp.do_carla_gen3dvq:
        log_dir_ = os.path.join("logs_carla_gen3dvq", hyp.name)
    elif hyp.do_carla_precompute:
        log_dir_ = os.path.join("logs_carla_precompute", hyp.name)
    elif hyp.do_carla_propose:
        
        # ## this dir is trying to learn some ok detnets:
        # log_dir_ = os.path.join("logs_carla_propose", hyp.name)

        ## this dir is for doing the 5-crit eval
        # log_dir_ = os.path.join("logs_carla_propose2", hyp.name)

        ## aws
        log_dir_ = os.path.join("logs_carla_propose", hyp.name)
        
    elif hyp.do_carla_det:
        log_dir_ = os.path.join("logs_carla_det", hyp.name)
    elif hyp.do_intphys_det:
        log_dir_ = os.path.join("logs_intphys_det", hyp.name)
    elif hyp.do_intphys_forecast:
        log_dir_ = os.path.join("logs_intphys_forecast", hyp.name)
    elif hyp.do_carla_forecast:
        log_dir_ = os.path.join("logs_carla_forecast", hyp.name)
    elif hyp.do_carla_pipe:
        log_dir_ = os.path.join("logs_carla_pipe", hyp.name)
    elif hyp.do_intphys_test:
        log_dir_ = os.path.join("logs_intphys_test", hyp.name)
    elif hyp.do_carla_pwc:
        log_dir_ = os.path.join("logs_carla_pwc", hyp.name)
    else:
        assert(False) # what mode is this?

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    try:
        if hyp.do_carla_static:
            model = CARLA_STATIC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_flo:
            model = CARLA_FLO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_time:
            model = CARLA_TIME(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_reloc:
            model = CARLA_RELOC(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_obj:
            model = CARLA_OBJ(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_focus:
            model = CARLA_FOCUS(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_track:
            model = CARLA_TRACK(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_siamese:
            model = CARLA_SIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_vsiamese:
            model = CARLA_VSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_rsiamese:
            model = CARLA_RSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_msiamese:
            model = CARLA_MSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ssiamese:
            model = CARLA_SSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_csiamese:
            model = CARLA_CSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_genocc:
            model = CARLA_GENOCC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_gengray:
            model = CARLA_GENGRAY(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_vqrgb:
            model = CARLA_VQRGB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_vq3drgb:
            model = CARLA_VQ3DRGB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_moc:
            model = CARLA_MOC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_don:
            model = CARLA_DON(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_don:
            model = KITTI_DON(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_moc:
            model = KITTI_MOC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_zoom:
            model = CARLA_ZOOM(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_zoom:
            model = KITTI_ZOOM(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_siamese:
            model = KITTI_SIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ml:
            model = CARLA_ML(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_bottle:
            model = CARLA_BOTTLE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_pretty:
            model = CARLA_PRETTY(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_bench:
            model = CARLA_BENCH(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_reject:
            model = CARLA_REJECT(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_auto:
            model = CARLA_AUTO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ret:
            model = CARLA_RET(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_clevr_vq3drgb:
            model = CLEVR_VQ3DRGB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_clevr_gen3dvq:
            model = CLEVR_GEN3DVQ(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_gen3dvq:
            model = CARLA_GEN3DVQ(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_precompute:
            model = CARLA_PRECOMPUTE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_propose:
            model = CARLA_PROPOSE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_det:
            model = CARLA_DET(checkpoint_dir=checkpoint_dir_, log_dir=log_dir_)
            model.go()
        elif hyp.do_intphys_det:
            model = INTPHYS_DET(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_intphys_forecast:
            model = INTPHYS_FORECAST(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_forecast:
            model = CARLA_FORECAST(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_pipe:
            model = CARLA_PIPE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_intphys_test:
            model = INTPHYS_TEST(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_pwc:
            model = CARLA_PWC(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        else:
            assert(False) # what mode is this?

    except (Exception, KeyboardInterrupt) as ex:
        logger.error(ex, exc_info=True)
        log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')

