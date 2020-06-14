from train import *

def train_yolo6D(datafile,cfgfile,weights):
    datacfg = datafile
    cfgfile = cfgfile
    weightfile = weights

    # datacfg = 'cfg/ape.data'
    # cfgfile = 'cfg/yolo-pose.cfg'
    # weightfile = 'backup/ape/init.weights'

    # Parse configuration files
    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]
    trainlist     = data_options['train']
    testlist      = data_options['valid']
    nsamples      = file_lines(trainlist)
    gpus          = data_options['gpus']  # e.g. 0,1,2,3
    gpus 		  = '0'
    meshname      = data_options['mesh']
    num_workers   = int(data_options['num_workers'])
    backupdir     = data_options['backup']
    diam          = float(data_options['diam'])
    vx_threshold  = diam * 0.1
    if not os.path.exists(backupdir):
        makedirs(backupdir)
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    bg_file_names = get_all_files('VOCdevkit/VOC2012/JPEGImages')

    # Train parameters
    max_epochs    = 700 # max_batches*batch_size/nsamples+1
    use_cuda      = True
    seed          = int(time.time())
    eps           = 1e-5
    save_interval = 10 # epoches
    dot_interval  = 70 # batches
    best_acc      = -1

    # Test parameters
    conf_thresh   = 0.1
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    im_width      = 640
    im_height     = 480

    # Specify which gpus to use
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Specifiy the model and the loss
    model       = Darknet(cfgfile)
    region_loss = model.loss

    # Model settings
    # model.load_weights(weightfile)
    model.load_weights_until_last(weightfile)
    model.print_network()
    model.seen = 0
    region_loss.iter  = model.iter
    region_loss.seen  = model.seen
    processed_batches = model.seen//batch_size
    init_width        = model.width
    init_height       = model.height
    test_width        = 672
    test_height       = 672
    init_epoch        = model.seen//nsamples

    # Variable to save
    training_iters          = []
    training_losses         = []
    testing_iters           = []
    testing_losses          = []
    testing_errors_trans    = []
    testing_errors_angle    = []
    testing_errors_pixel    = []
    testing_accuracies      = []

    # Get the intrinsic camerea matrix, mesh, vertices and corners of the model
    mesh                 = MeshPly(meshname)
    vertices             = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D            = get_3D_corners(vertices)
    internal_calibration = get_camera_intrinsic()

    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    # Get the dataloader for test data
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(testlist,
    															  shape=(test_width, test_height),
                                                                  shuffle=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(),]),
                                                                  train=False),
                                             batch_size=1, shuffle=False, **kwargs)

    # Pass the model to GPU
    if use_cuda:
        model = model.cuda() # model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # Multiple GPU parallelism

    # Get the optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimization

    evaluate = False
    if evaluate:
        logging('evaluating ...')
        test(0, 0)
    else:
        for epoch in range(init_epoch, max_epochs):
            # TRAIN
            niter = train(epoch)
            # TEST and SAVE
            if (epoch % 10 == 0) and (epoch is not 0):
                test(epoch, niter)
                logging('save training stats to %s/costs.npz' % (backupdir))
                np.savez(os.path.join(backupdir, "costs.npz"),
                    training_iters=training_iters,
                    training_losses=training_losses,
                    testing_iters=testing_iters,
                    testing_accuracies=testing_accuracies,
                    testing_errors_pixel=testing_errors_pixel,
                    testing_errors_angle=testing_errors_angle)
                if (testing_accuracies[-1] > best_acc ):
                    best_acc = testing_accuracies[-1]
                    logging('best model so far!')
                    logging('save weights to %s/model.weights' % (backupdir))
                    model.save_weights('%s/model.weights' % (backupdir))
        shutil.copy2('%s/model.weights' % (backupdir), '%s/model_backup.weights' % (backupdir))

if __name__ == "__main__":
    train_yolo6D('cfg\mangguo.data','yolo-pose.cfg','backup\init.weights')