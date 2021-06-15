using Cairo, Gtk

RIGHT_ARROW = 65363
LEFT_ARROW = 65361

function show_cairo(snapshot::T) where T <: CairoSurface
  show_cairo([snapshot])
end

function show_cairo(snapshots::AbstractVector{<: CairoSurface})
  snapshot = snapshots[1]
  win = Gtk.Window("Win")
  w, h = (snapshot.width, snapshot.height)
  Gtk.resize!(win, Int(w), Int(h))

  can = Gtk.Canvas(w, h)
  push!(win, can)

  function render_snapshot_i(idx)
    draw(can) do _
      ctx = getgc(can)
      rectangle(ctx, 0, 0, w, h)
      set_source_rgb(ctx, 1, 1, 1)
      set_source_surface(ctx, snapshots[idx])
      fill(ctx)
    end
    show(can)
    return
  end

  idx = 1

  function handle_keypress(widget, event)
    if event.keyval == Int('q')
      destroy(win)
    elseif event.keyval == RIGHT_ARROW
      idx =  min(idx + 1, length(snapshots))
      render_snapshot_i(idx)
    elseif event.keyval == LEFT_ARROW
      idx = max(idx - 1, 1)
      render_snapshot_i(idx)
    else
      println("Unrecognized key: ", event.keyval)
    end
  end
  signal_connect(handle_keypress, win, "key-press-event")

  render_snapshot_i(idx)
  return
end

function render_frames(scenes::AbstractVector{<:Scene}, entities...)
  images = []
  for scene in scenes
    car_colors = get_pastel_car_colors(scene)
    renders = vcat(
      Any[],
      collect(entities),
      [FancyCar(car = veh, color = car_colors[veh.id]) for veh in scene],
    )
    frame = render(renders)
    io = IOBuffer()
    write_to_png(frame, io)
    push!(images, load(io))
  end
  return reshape(
    reduce(hcat, map(image -> reshape(image, :), images)),
    size(images[1])...,
    :,
  )
end

function render_snapshots(scenes::AbstractVector{<:Scene}, entities...)
  snapshots = CairoSurface[]
  for scene in scenes
    car_colors = get_pastel_car_colors(scene)
    renders = vcat(
      Any[],
      collect(entities),
      [FancyCar(car = veh, color = car_colors[veh.id]) for veh in scene],
    )
    push!(snapshots, render(renders))
  end
  return snapshots
end
