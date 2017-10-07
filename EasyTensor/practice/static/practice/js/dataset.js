function Dataset() {
	var dim;
	var channels;
	var classes;
	var rows_per_batch;
	var num_batches;	
	var fully_loaded;
	var loading_batch;
	var batch_idx, t_batch_idx;
	var test_batches;
	var test_batch_only;
	var sample_idx = {train:0, test:0};
	var data = {train:[], test:[]};
	var callback_main = null;

	this.get_dim = function() {
		return dim;
	};

	this.get_channels = function() {
		return channels;
	};

	this.get_classes = function() {
		return classes;
	};

	this.get_training_size = function() {
		return data.train.length;
	};

	this.get_test_size = function() {
		return data.test.length;
	};

	this.get_training_size_all = function() {
		return (num_batches - test_batches.length) * rows_per_batch;
	};

	this.get_test_size_all = function() {
		return (num_batches - test_batches.length) * rows_per_batch;
	};

	this.is_loading = function() {
		return loading_batch;
	};

	this.is_fully_loaded = function() {
		return fully_loaded;
	};

	this.finished_testing = function() {
		return sample_idx.test >= test_batches.length * rows_per_batch;
	};

	this.get_training_sample = function(t) {
		return data.train[t];
	};

	this.get_test_sample = function(t) {
		return data.test[t];
	};

	this.get_sample_index = function() {
		return sample_idx;
	};

	this.get_next_training_sample = function() {
		if (sample_idx.train < data.train.length) {
			sample_idx.train += 1;
			return data.train[sample_idx.train-1];
		}
		else {
			return null;
		}
	};

	this.get_next_test_sample = function() {
		if (sample_idx.test < data.test.length) {
			return data.test[sample_idx.test++];
		} else {
			return null;
		}
	};

	this.set_train_index = function(idx) {
		sample_idx.train = idx;
	};

	this.set_test_index = function(idx) {
		sample_idx.test = idx;
	};

	this.initialize = function() {
		loading_batch = false;
		fully_loaded = false;
		sample_idx.train = 0;
		sample_idx.test = 0;
		t_batch_idx = 0;
		batch_idx = test_batch_only ? test_batches[t_batch_idx] : 0;
		this.load_batch(batch_idx);
	};

	this.loadMNIST = function(test_batch_only_, callback_main_) {
		if (callback_main_ != null) {
			callback_main = callback_main_;
		}
		test_batch_only = test_batch_only_;
		root_dir = '/static/practice/images/mnist/mnist';
		dim = 28;
		channels = 1;
		rows_per_batch = 3000;
		num_batches = 21;
		test_batches = [20];
		classes = ["0","1","2","3","4","5","6","7","8","9"];
		this.initialize();
	};
	
	this.load_batch = function(batch_idx, callback) {
		var batch_path = root_dir+"_batch_"+batch_idx+".png";
		loadImage(batch_path, function(img) {
			img.loadPixels();
			for (var r = 0; r < rows_per_batch; r++) {
		    	var b_label = labels[rows_per_batch * batch_idx + r];
		    	var b_vol = new convnetjs.Vol(dim, dim, channels, 0.0);
		    	var W = dim * dim;
		    	for (var i = 0; i < W; i++) {
		     		var ix = ((W * r) + i) * 4;
			      for (var c = 0; c < channels; c++) {
				     	b_vol.w[channels*i+c] = img.pixels[ix+c] / 255.0;
				     	// b_vol.w[channels*i+c] = img.pixels[ix+c];
						}
		    	}
		    	if (test_batches.indexOf(batch_idx) == -1) {
			    	data.train.push({idx:data.train.length, vol:b_vol, label:b_label});
			    } else {
			    	data.test.push({idx:data.test.length, vol:b_vol, label:b_label});	
			    }
			}
			console.log("loaded batch "+batch_idx+". size: {training set:"+data.train.length+", test set:"+data.test.length+")");
			fully_loaded = ((data.train.length + data.test.length) == (test_batch_only?test_batches.length:num_batches) * rows_per_batch);	
			loading_batch = false;
			if (callback != null) {
				callback();
			}
			if (callback_main != null && fully_loaded) {
				callback_main();
			}
		});
	};

	this.request_next_batch = function(callback){
		if (!loading_batch) {
			if (test_batch_only && t_batch_idx < test_batches.length-1) {
				t_batch_idx += 1;
				batch_idx = test_batches[t_batch_idx];
				loading_batch = true;
				this.load_batch(batch_idx, callback);
			}
			else if (batch_idx < num_batches-1) {
				batch_idx += 1;
				loading_batch = true;
				this.load_batch(batch_idx, callback);
			}
		}
	};

	this.get_image = function(sample) {
		var img = createImage(dim, dim);
		img.loadPixels();
		for (var i = 0; i< dim * dim; i++) {
			for (var j = 0; j < 3; j++) {
				// black based -> white based image
				img.pixels[4*i+j] = 255 * (1- sample.vol.w[i]);
			}
			img.pixels[4*i+3] = 255;
		}
		img.updatePixels();
		return img;
	};
}
